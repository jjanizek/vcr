import torch
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, pipeline
import requests
from typing import List, Dict, Union, Optional, Tuple
import os

class MedGemmaAPI:
    def __init__(
        self,
        model_name='MedGemma-4B-IT',
        hf_token=None,
    ):
        """
        Initialize MedGemma model
        
        Args:
            model_name: Model variant to use. Options:
                - 'MedGemma-4B-IT': 4B instruction-tuned multimodal
                - 'MedGemma-4B-PT': 4B pre-trained multimodal
                - 'MedGemma-27B-Multimodal': 27B multimodal instruction-tuned
                - 'MedGemma-27B-Text': 27B text-only instruction-tuned
            hf_token: Hugging Face authentication token (required for MedGemma models)
                     Get yours at https://huggingface.co/settings/tokens
                     Or use: huggingface-cli login
        """
        
        valid_models = {
            'MedGemma-4B-IT', 'MedGemma-4B-PT', 
            'MedGemma-27B-Multimodal', 'MedGemma-27B-Text'
        }
        assert model_name in valid_models, f"Error: Model '{model_name}' is not implemented. Valid models are: {', '.join(valid_models)}"
        
        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        self.model_name = model_name
        
        # Model ID mapping
        model_id_map = {
            'MedGemma-4B-IT': 'google/medgemma-4b-it',
            'MedGemma-4B-PT': 'google/medgemma-4b-pt',
            'MedGemma-27B-Multimodal': 'google/medgemma-27b-multimodal',
            'MedGemma-27B-Text': 'google/medgemma-27b-text'
        }
        
        self.model_id = model_id_map[model_name]
        self.is_text_only = 'Text' in model_name
        self.hf_token = hf_token
        
        # Check for authentication
        if self.hf_token is None:
            # Try to get token from environment variable
            self.hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
            
            if self.hf_token is None:
                print("\n" + "="*70)
                print("MedGemma requires authentication!")
                print("="*70)
                print("\nTo use MedGemma models, you need to:")
                print("1. Request access at: https://huggingface.co/google/medgemma-4b-it")
                print("2. Get your token at: https://huggingface.co/settings/tokens")
                print("3. Either:")
                print("   a) Pass token: MedGemmaAPI(hf_token='your_token_here')")
                print("   b) Set environment: export HF_TOKEN='your_token_here'")
                print("   c) Login via CLI: huggingface-cli login")
                print("="*70 + "\n")
                raise ValueError("Hugging Face token required for MedGemma models")
        
        try:
            # Try using pipeline API first (most compatible)
            if not self.is_text_only:
                self.pipeline = pipeline(
                    "image-text-to-text",
                    model=self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    use_auth_token=self.hf_token,  # Use older parameter name
                )
                # Extract model and processor from pipeline
                self.model = self.pipeline.model
                self.processor = self.pipeline.processor
                self.tokenizer = self.pipeline.tokenizer
            else:
                # For text-only model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    use_auth_token=self.hf_token,  # Use older parameter name
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_auth_token=self.hf_token  # Use older parameter name
                )
                self.processor = None
                self.pipeline = None
                
        except Exception as e:
            print(f"Warning: Could not load with pipeline, trying alternative approach: {e}")
            # Fallback: Try loading with AutoModelForCausalLM for all models
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,  # Some models may need this
                    use_auth_token=self.hf_token,  # Use older parameter name
                )
                
                # Try to load processor, fall back to tokenizer if not available
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_id,
                        use_auth_token=self.hf_token  # Use older parameter name
                    )
                    self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None
                except:
                    self.processor = None
                    
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id,
                        use_auth_token=self.hf_token  # Use older parameter name
                    )
                    
                self.pipeline = None
                    
            except Exception as e2:
                print(f"Error loading model: {e2}")
                print("Please ensure you have the latest transformers version: pip install -U transformers>=4.50.0")
                raise
        
        # Configure tokenizer
        if self.tokenizer and hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = "left"
        if self.tokenizer and (not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
    
    def load_and_resize_image(self, path: Union[str, Image.Image], max_size: int = 600) -> Image.Image:
        """Load and resize image while maintaining aspect ratio"""
        try:
            if isinstance(path, str):
                if path.startswith('http://') or path.startswith('https://'):
                    # Load from URL
                    img = Image.open(requests.get(path, headers={"User-Agent": "MedGemmaAPI"}, stream=True).raw)
                else:
                    # Load from local file
                    img = Image.open(path)
            elif isinstance(path, Image.Image):
                img = path
            else:
                raise ValueError(f"Unsupported image type: {type(path)}")
            
            img = img.convert('RGB')
            
            if img.size[0] > max_size or img.size[1] > max_size:
                ratio = min(max_size/img.size[0], max_size/img.size[1])
                new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            return img
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            raise
    
    def preprocess_images(self, images: List[Union[str, Image.Image]]) -> List[Image.Image]:
        """Preprocess a list of images"""
        if self.is_text_only:
            raise ValueError("Text-only model does not support image processing")
        
        try:
            processed_images = []
            for image in images:
                img = self.load_and_resize_image(image)
                processed_images.append(img)
            return processed_images
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def get_choice_logprobs(self, prompt: str, choices: List[str], image_paths: List[Union[str, Image.Image]] = []) -> Dict[str, float]:
        """Calculate log probabilities for each choice"""
        if self.is_text_only and image_paths:
            raise ValueError("Text-only model does not support images")
        
        choice_logprobs = {}
        
        for choice in choices:
            full_text = f"{prompt} {choice}"
            
            if self.is_text_only or not image_paths:
                # Text-only processing
                encoded = self.tokenizer(full_text, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            else:
                # Multimodal processing
                if self.pipeline:
                    # Use pipeline for multimodal if available
                    images = self.preprocess_images(image_paths)
                    
                    messages = [
                        {
                            "role": "user",
                            "content": []
                        }
                    ]
                    
                    messages[0]["content"].append({"type": "text", "text": full_text})
                    for img in images:
                        messages[0]["content"].append({"type": "image", "image": img})
                    
                    # Get model inputs through pipeline processing
                    if hasattr(self.processor, 'apply_chat_template'):
                        inputs = self.processor.apply_chat_template(
                            messages, 
                            add_generation_prompt=False, 
                            tokenize=True,
                            return_dict=True, 
                            return_tensors="pt"
                        ).to(self.device)
                    else:
                        # Fallback for older versions
                        text_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
                        inputs = text_inputs.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                else:
                    # Fallback to text-only processing if pipeline not available
                    print("Warning: Processing as text-only due to missing multimodal support")
                    encoded = self.tokenizer(full_text, padding=True, truncation=True, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded["attention_mask"].to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
            
            # Calculate log probabilities
            logits = outputs.logits[0]
            
            # Tokenize prompt separately to find where choice starts
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            
            if self.is_text_only or not image_paths:
                full_tokens = input_ids[0]
            else:
                full_tokens = inputs["input_ids"][0] if "input_ids" in inputs else input_ids[0]
            
            choice_start = len(prompt_tokens) - 1
            
            choice_logprob = 0
            for idx in range(choice_start, min(len(full_tokens) - 1, len(logits) - 1)):
                token_logits = logits[idx]
                next_token = full_tokens[idx + 1]
                token_probs = torch.log_softmax(token_logits, dim=-1)
                if next_token < len(token_probs):
                    choice_logprob += token_probs[next_token].item()
            
            choice_logprobs[choice] = choice_logprob
        
        return choice_logprobs
    
    def get_best_choice(self, prompt: str, choices: List[str], image_paths: List[Union[str, Image.Image]] = []) -> Tuple[str, Dict[str, float]]:
        """Get the most likely choice based on log probabilities"""
        logprobs = self.get_choice_logprobs(prompt, choices, image_paths)
        best_choice = max(logprobs.items(), key=lambda x: x[1])[0]
        return best_choice, logprobs
    
    def __call__(
        self, 
        prompt: str, 
        image_paths: List[Union[str, Image.Image]] = [], 
        max_new_tokens: int = 200,
        num_beams: int = 1,
        min_length: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text response given prompt and optional images
        
        Args:
            prompt: Text prompt
            image_paths: List of image paths or PIL Images (for multimodal models)
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            min_length: Minimum length of generated sequence
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            system_prompt: Optional system prompt (e.g., "You are an expert radiologist.")
        
        Returns:
            Generated text response
        """
        if self.is_text_only and image_paths:
            raise ValueError("Text-only model does not support images")
        
        if not isinstance(image_paths, list):
            image_paths = [image_paths] if image_paths else []
        
        # Use pipeline if available for multimodal
        if self.pipeline and image_paths and not self.is_text_only:
            images = self.preprocess_images(image_paths)
            
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            user_content = [{"type": "text", "text": prompt}]
            for img in images:
                user_content.append({"type": "image", "image": img})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Use pipeline for generation
            output = self.pipeline(
                text=messages,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
            )
            
            # Extract generated text from pipeline output
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], dict):
                    if 'generated_text' in output[0]:
                        generated = output[0]['generated_text']
                        # If it's a conversation, get the last message
                        if isinstance(generated, list) and len(generated) > 0:
                            last_message = generated[-1]
                            if isinstance(last_message, dict) and 'content' in last_message:
                                return last_message['content'].strip()
                        return str(generated).strip()
            
            return str(output).strip()
        
        else:
            # Fallback to direct model generation
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Prepare inputs
            encoded = self.tokenizer(full_prompt, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    temperature=temperature if do_sample else 1.0,
                    do_sample=do_sample,
                    top_k=top_k if do_sample else None,
                    top_p=top_p if do_sample else None,
                    min_length=min_length,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode only the new tokens
            input_length = input_ids.shape[1]
            new_tokens = generated[0][input_length:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()