import torch
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import requests
from typing import List, Dict, Union, Optional, Tuple
import os
import warnings

class LLaVaMedAPI:
    def __init__(
        self,
        model_name='LLaVA-Med-7B',
        load_in_8bit=False,
        load_in_4bit=False,
    ):
        """
        Initialize LLaVA-Med model
        
        Args:
            model_name: Model variant to use. Options:
                - 'LLaVA-Med-7B': LLaVA-Med v1.5 Mistral 7B (default)
                - 'LLaVA-Med-13B': LLaVA-Med v1.0 13B 
            load_in_8bit: Whether to load model in 8-bit precision (saves memory)
            load_in_4bit: Whether to load model in 4-bit precision (saves even more memory)
        """
        
        valid_models = {
            'LLaVA-Med-7B', 'LLaVA-Med-13B'
        }
        assert model_name in valid_models, f"Error: Model '{model_name}' is not implemented. Valid models are: {', '.join(valid_models)}"
        
        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device if not (load_in_8bit or load_in_4bit) else None
        self.model_name = model_name
        
        # Model ID mapping
        model_id_map = {
            'LLaVA-Med-7B': 'microsoft/llava-med-v1.5-mistral-7b',
            'LLaVA-Med-13B': 'microsoft/llava-med-7b',  # Note: This is the v1.0 model
        }
        
        self.model_id = model_id_map[model_name]
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        try:
            print(f"Loading LLaVA-Med model: {self.model_id}")
            print("This may take a while for the first download...")
            
            # Try to load with LlavaForConditionalGeneration first
            try:
                from transformers import LlavaForConditionalGeneration
                
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if not (load_in_8bit or load_in_4bit) else None,
                    device_map="auto",
                    quantization_config=quantization_config,
                )
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.tokenizer = self.processor.tokenizer
                print("Successfully loaded with LlavaForConditionalGeneration")
                
            except ImportError:
                print("LlavaForConditionalGeneration not found, trying AutoModelForCausalLM...")
                # Fallback to AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if not (load_in_8bit or load_in_4bit) else None,
                    device_map="auto",
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                )
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else AutoTokenizer.from_pretrained(self.model_id)
                print("Successfully loaded with AutoModelForCausalLM")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying alternative loading approach...")
            
            # Final fallback
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                )
                
                try:
                    self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                    self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None
                except:
                    self.processor = None
                    
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                    
            except Exception as e2:
                print(f"Final error: {e2}")
                print("\n" + "="*70)
                print("Installation requirements for LLaVA-Med:")
                print("="*70)
                print("pip install transformers>=4.28.0")
                print("pip install bitsandbytes  # For 8-bit/4-bit loading")
                print("pip install accelerate")
                print("="*70 + "\n")
                raise
        
        # Configure tokenizer
        if hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = "left"
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Only prepare model if not using quantization
        if not (load_in_8bit or load_in_4bit):
            self.model = self.accelerator.prepare(self.model)
        
        self.model.eval()
        print(f"Model loaded successfully!")
    
    def load_and_resize_image(self, path: Union[str, Image.Image], max_size: int = 336) -> Image.Image:
        """Load and resize image while maintaining aspect ratio
        Note: LLaVA typically uses 336x336 images"""
        try:
            if isinstance(path, str):
                if path.startswith('http://') or path.startswith('https://'):
                    # Load from URL
                    img = Image.open(requests.get(path, headers={"User-Agent": "LLaVaMedAPI"}, stream=True).raw)
                else:
                    # Load from local file
                    img = Image.open(path)
            elif isinstance(path, Image.Image):
                img = path
            else:
                raise ValueError(f"Unsupported image type: {type(path)}")
            
            img = img.convert('RGB')
            
            # LLaVA uses specific image sizes, typically 336x336
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
        try:
            processed_images = []
            for image in images:
                img = self.load_and_resize_image(image)
                processed_images.append(img)
            return processed_images
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for LLaVA-Med"""
        # LLaVA-Med uses a specific prompt format
        if system_prompt:
            formatted = f"{system_prompt}\n\nUSER: {prompt}\nASSISTANT:"
        else:
            formatted = f"USER: {prompt}\nASSISTANT:"
        return formatted
    
    def get_choice_logprobs(self, prompt: str, choices: List[str], image_paths: List[Union[str, Image.Image]] = []) -> Dict[str, float]:
        """Calculate log probabilities for each choice"""
        
        choice_logprobs = {}
        images = self.preprocess_images(image_paths) if image_paths else []
        
        for choice in choices:
            full_text = f"{prompt} {choice}"
            formatted_text = self.format_prompt(full_text)
            
            if images and self.processor:
                # Process with images
                inputs = self.processor(
                    text=formatted_text,
                    images=images[0] if len(images) == 1 else images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                if self.device:
                    inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            else:
                # Text-only processing
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                if self.device:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Calculate log probabilities
            logits = outputs.logits[0]
            
            # Find where choice starts
            prompt_tokens = self.tokenizer.encode(self.format_prompt(prompt), add_special_tokens=True)
            full_tokens = inputs["input_ids"][0]
            
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
        temperature: float = 0.7,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.95,
        system_prompt: Optional[str] = "You are a helpful medical AI assistant."
    ) -> str:
        """
        Generate text response given prompt and optional images
        
        Args:
            prompt: Text prompt
            image_paths: List of image paths or PIL Images
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            min_length: Minimum length of generated sequence
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            system_prompt: Optional system prompt for medical context
        
        Returns:
            Generated text response
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths] if image_paths else []
        
        images = self.preprocess_images(image_paths) if image_paths else []
        formatted_prompt = self.format_prompt(prompt, system_prompt)
        
        # Process inputs
        if images and self.processor:
            inputs = self.processor(
                text=formatted_prompt,
                images=images[0] if len(images) == 1 else images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        else:
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        # Move to device if available
        if self.device:
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
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
                no_repeat_ngram_size=3
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Clean up the response (remove any remaining prompt artifacts)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response