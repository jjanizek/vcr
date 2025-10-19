import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.prompt_template = "Is the lesion benign or malignant? <image> The lesion is"
       
    def __len__(self):
        return len(self.image_paths)
   
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return {
            'image': self.preprocess(image),
            'prompt': self.prompt_template,
            'image_path': str(self.image_paths[idx]),
            'label': ''
        }
        
class PathDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
       
    def __len__(self):
        return len(self.image_paths)
   
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.preprocess(image)

class CLIPEmbedder:
    def __init__(self, model_name='ViT-L-14', pretrained='openai', batch_size=32):
        self.model, self.preprocess = create_model_from_pretrained(model_name, pretrained=pretrained)
        self.tokenizer = get_tokenizer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def get_image_embeddings(self, image_paths):
        dataset = PathDataset(image_paths, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        
        embeddings = []
        for batch in tqdm(dataloader, desc="Getting image embeddings"):
            batch = batch.to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(batch)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings)

    def get_text_embeddings(self, texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Getting text embeddings"):
            batch = texts[i:i + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_text(tokens)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings)

class AttentionExtractor:
    def __init__(self, model):
        self.model = model.model.cuda()
        self.tokenizer = model.tokenizer
        self.image_processor = model.image_processor
        self.activations = {}
        
    def save_activation(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                attention_output = output[0]
            else:
                attention_output = output
            self.activations[name] = attention_output.detach().cpu()
        return hook
        
    def get_attention_activations(self, prompt, image_batch):
        """Return a dict containing:
           - last_text_token: final-layer hidden states for the last non-PAD text token
           - last_image_token: final-layer hidden states for the last <image> token
        """
        # Reset old activations
        self.activations = {}
        
        # Register hook on the final attention layer
        self.model.lang_encoder.transformer.blocks[23].decoder_layer.attn.register_forward_hook(
            self.save_activation('final_attention')
        )
        
        # Ensure shape is (batch, T_img, frames, channels, H, W)
        if len(image_batch.shape) == 4:
            image_batch = image_batch.unsqueeze(1).unsqueeze(2)
        
        image_batch = image_batch.cuda()
        
        # Repeat the prompt for each image in the batch
        batch_size = image_batch.shape[0]
        prompts = [prompt] * batch_size
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()
        
        with torch.no_grad():
            self.model(
                vision_x=image_batch,
                lang_x=input_ids,
                attention_mask=attention_mask,
            )
        
        # final_attention: shape [batch_size, seq_len, d_model]
        final_attention = self.activations["final_attention"]
        
        # We'll build lists of the hidden states we want for each example
        last_text_hidden = []
        last_image_hidden = []
        
        pad_id = self.tokenizer.pad_token_id
        
        for i in range(batch_size):
            # 1) Last text token index = number of non-pad tokens - 1
            seq_len_no_pad = (input_ids[i] != pad_id).sum().item()
            last_text_idx = seq_len_no_pad - 1
            
            # 2) Last <image> token index
            #    Convert this single example's IDs to tokens
            token_ids_i = input_ids[i].tolist()
            token_strings_i = self.tokenizer.convert_ids_to_tokens(token_ids_i)
            
            # Find all indices of <image> in this sequence
            image_indices = [idx for idx, token in enumerate(token_strings_i) if token == "<image>"]
            
            if len(image_indices) > 0:
                last_image_idx = image_indices[-1]
            else:
                # If there's no <image> token at all, let's just store None or skip
                last_image_idx = None
            
            # Gather hidden states
            last_text_acts = final_attention[i, last_text_idx, :]  # shape [d_model]
            if last_image_idx is not None:
                last_image_acts = final_attention[i, last_image_idx, :]  # shape [d_model]
            else:
                # e.g. fill with zeros or skip
                last_image_acts = torch.zeros_like(last_text_acts)
            
            last_text_hidden.append(last_text_acts)
            last_image_hidden.append(last_image_acts)
        
        # Stack into shape [batch_size, d_model]
        last_text_hidden = torch.stack(last_text_hidden, dim=0)
        last_image_hidden = torch.stack(last_image_hidden, dim=0)
        
        return {
            "last_text_token": last_text_hidden, 
            "last_image_token": last_image_hidden
        }

def compute_inner_products(text_embeddings, image_embeddings):
    """Compute similarity matrix between text and image embeddings"""
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    return torch.matmul(text_embeddings, image_embeddings.t())

class LayerOverride(nn.Module):
    """Generic layer override class that can be used for any module"""
    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module
        self.override = None
        self._forward_func = None  # Store the module's forward function
        
        # Detect the module type and store the appropriate forward method
        if hasattr(original_module, 'forward'):
            self._forward_func = original_module.forward
        
    def forward(self, *args, **kwargs):
        """Forward method that handles overriding for different module types"""
        if self.override is not None:
            # Get original output
            with torch.no_grad():
                orig_output = self._forward_func(*args, **kwargs)
            
            # Handle different return types (single tensor, tuple, etc.)
            if isinstance(orig_output, tuple):
                # For modules that return multiple values (like attention)
                new_output = list(orig_output)
                if isinstance(new_output[0], torch.Tensor):
                    # Verify shapes match exactly
                    if new_output[0].shape != self.override.shape:
                        raise ValueError(f"Shape mismatch: original output shape {new_output[0].shape} vs override shape {self.override.shape}")
                    new_output[0] = new_output[0] * 0 + self.override  # Use exact replacement
                return tuple(new_output)
            elif isinstance(orig_output, torch.Tensor):
                # For modules that return a single tensor
                # Verify shapes match exactly
                if orig_output.shape != self.override.shape:
                    raise ValueError(f"Shape mismatch: original output shape {orig_output.shape} vs override shape {self.override.shape}")
                return orig_output * 0 + self.override  # Use exact replacement
            else:
                # Fallback to original output for unsupported types
                return orig_output
                
        return self._forward_func(*args, **kwargs)