import torch
from PIL import Image
from accelerate import Accelerator
from einops import repeat
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
import os
import json

class FlamingoAPI:
    def __init__(
        self,
        model_name='OpenFlamingo-3B-Instruct',
    ):

        valid_models = {'OpenFlamingo-3B-Instruct', 'OpenFlamingo-4B', 'OpenFlamingo-9B', 'MedFlamingo'}
        assert model_name in valid_models, f"Error: Model '{model_name}' is not implemented. Valid models are: {', '.join(valid_models)}"
        
#         self.accelerator = Accelerator(cpu=True)
        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        
        init_dict = {
            'OpenFlamingo-3B-Instruct': {
                'checkpoint_path': "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
                'lang_encoder_path': "anas-awadalla/mpt-1b-redpajama-200b-dolly",
                'tokenizer_path': "anas-awadalla/mpt-1b-redpajama-200b-dolly",
                'cross_attn_every_n_layers': 1,
            },
            'OpenFlamingo-4B': {
                'checkpoint_path': "openflamingo/OpenFlamingo-4B-vitl-rpj3b",
                'lang_encoder_path': "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                'tokenizer_path': "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                'cross_attn_every_n_layers': 2,
            },
            'OpenFlamingo-9B': {
                'checkpoint_path': "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
                'lang_encoder_path': "anas-awadalla/mpt-7b",
                'tokenizer_path': "anas-awadalla/mpt-7b",
                'cross_attn_every_n_layers': 4,
            },
        }
        
        if model_name == 'MedFlamingo':
            # >>> add your local path to Llama-7B (v1) model here:
            llama_path = 'your/local/path/to/decapoda-research-llama-7B-hf'
            if not os.path.exists(llama_path):
                raise ValueError('Llama model not yet set up, please check README for instructions!')
            
            self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path=llama_path,
                tokenizer_path=llama_path,
                cross_attn_every_n_layers=4,
                use_local_files=True,
            )
            
            # load med-flamingo checkpoint:
            checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt", local_files_only=True, cache_dir="/your/path/to/the/cache/dir")
            print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
            
            self.tokenizer.padding_side = "left" 
        
        else: 
        
            # Initialize model components
            self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path=init_dict[model_name]['lang_encoder_path'],
                tokenizer_path=init_dict[model_name]['tokenizer_path'],
                cross_attn_every_n_layers=init_dict[model_name]['cross_attn_every_n_layers'],
                use_local_files=False,
            )
        
            # Load checkpoint and prepare model
            checkpoint = hf_hub_download(init_dict[model_name]['checkpoint_path'], "checkpoint.pt", local_files_only=True)
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device), strict=False)
            
            # Configure tokenizer
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
    
    def load_and_resize_image(self, path, max_size=600):
        """Load and resize image while maintaining aspect ratio"""
        try:
            img = Image.open(path).convert('RGB')
            if img.size[0] > max_size or img.size[1] > max_size:
                ratio = min(max_size/img.size[0], max_size/img.size[1])
                new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            raise
    
    def preprocess_images(self, images):
        """Preprocess a list of images"""
        try:
            pixels = []
            for image in images:
                if isinstance(image, str):
                    image = self.load_and_resize_image(image)
                pixels.append(self.image_processor(image))
            return torch.stack(pixels)
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def get_choice_logprobs(self, prompt, choices, image_paths=[]):
        """Calculate log probabilities for each choice without using caching."""
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        # Process images
        pixels = self.preprocess_images([self.load_and_resize_image(img) for img in image_paths])
        pixels = pixels.to(self.device)
        pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)

        choice_logprobs = {}
        for choice in choices:
            full_text = f"{prompt} {choice}"
            encoded = self.tokenizer(full_text, padding=True, truncation=True, return_tensors="pt")

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    vision_x=pixels,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits[0]
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                choice_start = len(prompt_tokens) - 1

                choice_logprob = 0
                for idx in range(choice_start, len(input_ids[0])):
                    token_logits = logits[idx-1]
                    next_token = input_ids[0][idx]
                    token_probs = torch.log_softmax(token_logits, dim=-1)
                    choice_logprob += token_probs[next_token].item()

                choice_logprobs[choice] = choice_logprob

        return choice_logprobs

    def get_best_choice(self, prompt, choices, image_paths=[]):
        """Get the most likely choice based on log probabilities"""
        logprobs = self.get_choice_logprobs(prompt, choices, image_paths)
        return max(logprobs.items(), key=lambda x: x[1])[0], logprobs
    
    def __call__(self, prompt, image_paths=[], max_new_tokens=20, num_beams=3, min_length=3):
        """Generate text response given prompt and images"""
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        pixels = self.preprocess_images([self.load_and_resize_image(img) for img in image_paths])
        pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
        encoded_text = self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            generated_text = self.model.generate(
                vision_x=pixels.to(self.device),
                lang_x=encoded_text["input_ids"].to(self.device),
                attention_mask=encoded_text["attention_mask"].to(self.device),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=0.9,
                do_sample=True,
                top_k=100,
                top_p=0.9,
                min_length=min_length,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )

        input_length = encoded_text["input_ids"].shape[1]
        new_tokens = generated_text[0][input_length:]
#         return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return self.tokenizer.decode(generated_text[0])
