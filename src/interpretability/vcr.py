#!/usr/bin/env python
# vcr.py
"""
Core module for visual concept-based analysis of LMMs.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from einops import repeat
import re
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import torch.nn.functional as F
from models.flamingo import FlamingoAPI
from collections import defaultdict

class PromptTemplate:
    """Class for managing prompt templates."""
    
    def __init__(self, base_prompt, demo_template=None, query_template=None):
        """
        Initialize prompt template.
        
        Args:
            base_prompt: Base prompt string (can include {demo} placeholder)
            demo_template: Template for each demonstration
            query_template: Template for the query
        """
        self.base_prompt = base_prompt
        self.demo_template = demo_template or "<image>Based on the image, this lesion is {label}.<|endofchunk|>"
        self.query_template = query_template or "<image>Based on the image, this lesion is"
    
    def build_prompt(self, demo_labels=None):
        """Build the full prompt with optional demonstrations."""
        if demo_labels is None:
            # Zero-shot case
            return self.base_prompt + self.query_template
        else:
            # Few-shot case
            demo_prompts = ""
            for label in demo_labels:
                demo_prompts += self.demo_template.format(label=label)
            return self.base_prompt + demo_prompts + self.query_template


class ConceptAnalyzer:
    """Main class for concept-based model analysis."""
    
    def __init__(self, model_name, clip_embedder, image_processor=None):
        """
        Initialize the analyzer.
        
        Args:
            model: The vision-language model to analyze
            clip_embedder: CLIP embedder for computing concept embeddings
            image_processor: Optional image processor (will use model's if not provided)
        """
        
        self.model = FlamingoAPI(model_name=model_name)
        self.model_name = model_name
        self.clip = clip_embedder
        self.image_processor = image_processor or self.model.image_processor
        self.wrapped_layer = None
        self.concept_model = None
        self.concept_vectors = None
        
    def setup_layer_hook(self, target_layer_name, layer_wrapper_class):
        """
        Set up a hook on the specified layer.

        Args:
            target_layer_name: Dot-separated path to the layer
            layer_wrapper_class: Class to wrap the layer with (e.g., LayerOverride)

        Returns:
            The wrapped layer module
        """
        # Find the target layer
        parts = target_layer_name.split('.')

        # Start from the model object
        target_module = self.model

        # Check if we need to access the inner model first
        # This handles cases where the model is wrapped in FlamingoAPI
        if hasattr(self.model, 'model'):
            # Check if the first part exists in the wrapper or the inner model
            if hasattr(self.model, parts[0]):
                target_module = self.model
            elif hasattr(self.model.model, parts[0]):
                target_module = self.model.model
            else:
                # For debugging, let's see what attributes are available
                print(f"Available attributes in self.model: {dir(self.model)}")
                if hasattr(self.model, 'model'):
                    print(f"Available attributes in self.model.model: {dir(self.model.model)}")
                raise AttributeError(f"Cannot find {parts[0]} in model structure")

        # Extract block/layer number for debugging
        block_num = None
        if self.model_name == 'OpenFlamingo-3B-Instruct': 
            block_match = re.search(r'blocks\.(\d+)', target_layer_name)
            block_num = int(block_match.group(1)) if block_match else None
        elif self.model_name in ['MedFlamingo', 'OpenFlamingo-4B']:
            block_match = re.search(r'layers\.(\d+)', target_layer_name)
            block_num = int(block_match.group(1)) if block_match else None

        # Store the root module for later
        root_module = target_module

        # Traverse to the target module
        for part in parts:
            if part.isdigit():
                target_module = target_module[int(part)]
            else:
                target_module = getattr(target_module, part)

        # Wrap the layer
        self.wrapped_layer = layer_wrapper_class(target_module)

        # Set the wrapped module back in the model
        current_module = root_module

        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)

        last_part = parts[-1]
        if last_part.isdigit():
            current_module[int(last_part)] = self.wrapped_layer
        else:
            setattr(current_module, last_part, self.wrapped_layer)

        # Print what we've wrapped for debugging
        print(f"Wrapped {target_layer_name} (block {block_num})" if block_num is not None else f"Wrapped {target_layer_name}")

        return self.wrapped_layer

    def get_embeddings(self, image_paths, concept_files):
        """
        Compute CLIP embeddings for images and concepts.
        
        Args:
            image_paths: List of paths to images
            concept_files: List of text files containing concepts
            
        Returns:
            Tuple of (image_embeddings, text_embeddings, concept_texts)
        """
        # Load concept texts
        texts = []
        for f in concept_files:
            with open(f) as file:
                texts.extend(line.strip() for line in file)
        
        # Compute embeddings
        image_embeddings = self.clip.get_image_embeddings(image_paths)
        text_embeddings = self.clip.get_text_embeddings(texts)
        
        return image_embeddings, text_embeddings, texts
    
    def process_images_for_model(self, image_paths):
        """Process a list of images for model input."""
        processed_images = []
        for img_path in image_paths:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_images.append(self.image_processor(image))
        return processed_images
    
    def collect_activations(self, dataset, prompt_template, demo_paths=None, 
                          demo_labels=None, batch_size=1, num_workers=4):
        """
        Collect activations from the wrapped layer.
        
        Args:
            dataset: Dataset of images to process
            prompt_template: Template for generating prompts
            demo_paths: Optional paths to demonstration images for ICL
            demo_labels: Optional labels for demonstration images
            batch_size: Batch size for processing
            num_workers: Number of dataloader workers
            
        Returns:
            Tensor of collected activations
        """
        if self.wrapped_layer is None:
            raise RuntimeError("No layer wrapped. Call setup_layer_hook first.")
            
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                              num_workers=num_workers, shuffle=False)
        layer_outputs = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                layer_outputs.append(output[0].detach().cpu())
            else:
                layer_outputs.append(output.detach().cpu())
        
        hook = self.wrapped_layer.register_forward_hook(hook_fn)
        
        # Process demo images if provided
        stacked_demos = None
        if demo_paths is not None and demo_labels is not None:
            processed_imgs = self.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_imgs)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=batch_size)
        
        # Build prompt
        prompt = prompt_template.build_prompt(demo_labels if demo_labels else None)
        batch_prompt = [prompt] * batch_size
        
        self.model.model.eval()
        for batch in tqdm(dataloader, desc="Collecting activations"):
            image_batch = batch['image'].cuda()
            if len(image_batch.shape) == 4:
                image_batch = image_batch.unsqueeze(1).unsqueeze(2)
            
            if stacked_demos is not None:
                image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)
            
            encoded = self.model.tokenizer(batch_prompt, return_tensors="pt", 
                                         padding=True, truncation=True)
            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()
            
            outputs = self.model.model(vision_x=image_batch, lang_x=input_ids, 
                                     attention_mask=attention_mask)
        
        hook.remove()
        activations = torch.cat(layer_outputs, dim=0)
        # Add this debug code in collect_activations:
        print(f"Layer activation shape: {activations.shape}")
        print(f"Memory usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        return activations
    
    def train_concept_model(self, activations, similarity_matrix, 
                          test_size=0.2, alpha=1.0, random_state=42):
        """
        Train a model to predict concept similarities from activations.
        
        Args:
            activations: Tensor of layer activations [samples, features]
            similarity_matrix: Tensor of concept similarities [concepts, samples]
            test_size: Proportion of data to use for testing
            alpha: Ridge regression regularization parameter
            random_state: Random seed for train/test split
            
        Returns:
            Dictionary containing model, predictions, and metrics
        """
        from sklearn.model_selection import train_test_split
        
        # Use only the last token's activations
        if len(activations.shape) == 3:
            X = activations[:, -1, :].numpy()
        else:
            X = activations.numpy()
            
        Y = similarity_matrix.T.numpy()  # Transpose to [samples, concepts]
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.concept_model = Ridge(alpha=alpha)
        self.concept_model.fit(X_train, Y_train)
        
        # Evaluate
        Y_pred = self.concept_model.predict(X_test)
        
        # Calculate R² for each concept
        r2_per_concept = np.array([
            r2_score(Y_test[:, i], Y_pred[:, i]) 
            for i in range(Y_test.shape[1])
        ])
        
        return {
            'model': self.concept_model,
            'predictions': Y_pred,
            'y_test': Y_test,
            'r2_scores': r2_per_concept,
            'overall_r2': np.mean(r2_per_concept)
        }
        
    def extract_concept_vectors(self):
        """Extract and normalize concept vectors from the trained model."""
        if self.concept_model is None:
            raise RuntimeError("No concept model trained. Call train_concept_model first.")

        # Ridge with multiple outputs stores coefficients as a matrix
        coef_matrix = self.concept_model.coef_

        # For Ridge with multiple outputs, coef_ has shape (n_targets, n_features)
        if len(coef_matrix.shape) == 1:
            # Single output case
            concept_vectors = coef_matrix.reshape(1, -1)
        else:
            # Multiple output case: coef_matrix is already (n_concepts, n_features)
            concept_vectors = coef_matrix

        # Normalize each concept vector to unit length
        norms = np.linalg.norm(concept_vectors, axis=1, keepdims=True)
        concept_vectors = concept_vectors / norms

        self.concept_vectors = torch.tensor(concept_vectors, dtype=torch.float32)
        return self.concept_vectors
    
#     def extract_concept_vectors(self):
#         """Extract and normalize concept vectors from the trained model."""
#         if self.concept_model is None:
#             raise RuntimeError("No concept model trained. Call train_concept_model first.")
            
#         concept_vectors = []
#         for estimator in self.concept_model.estimators_:
#             vector = estimator.coef_
#             # Normalize to unit vector
#             vector = vector / np.linalg.norm(vector)
#             concept_vectors.append(vector)
            
#         self.concept_vectors = torch.tensor(np.array(concept_vectors), dtype=torch.float32)
#         return self.concept_vectors
    
    def compute_concept_weights(self, similarity_matrix, weight_type='variance'):
        """
        Compute importance weights for concepts.
        
        Args:
            similarity_matrix: Tensor of concept similarities [concepts, samples]
            weight_type: Type of weighting ('variance', 'uniform', etc.)
            
        Returns:
            Tensor of concept weights
        """
        Y = similarity_matrix.T.numpy()  # [samples, concepts]
        
        if weight_type == 'variance':
            weights = np.var(Y, axis=0)
        elif weight_type == 'uniform':
            weights = np.ones(Y.shape[1])
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
            
        return torch.tensor(weights, dtype=torch.float32)

    def compute_model_outputs(self, image_batch, prompt_batch, completion):
        """
        Compute the difference in log probabilities between choices.
        
        Args:
            image_batch: Tensor of images
            prompt_batch: List of prompts
            completion: Single completion string
            
        Returns:
            Tensor of choice differences
        """
        device = next(self.model.model.parameters()).device
        batch_size = len(prompt_batch)
        
        # Tokenize choices
        choice_ids = self.model.tokenizer.encode(completion, add_special_tokens=False) 
        
        # Initialize log probabilities
        choice_logprobs = torch.zeros(batch_size, 1, device=device)
        
        # Compute log probabilities for the completion
        full_texts = [f"{prompt}{completion}" for prompt in prompt_batch]
            
        # Tokenize
        encoded = self.model.tokenizer(
            full_texts, 
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
            
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
            
        # Get model outputs
        with torch.set_grad_enabled(True):
            outputs = self.model.model(
                vision_x=image_batch,
                lang_x=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
        # Calculate log probabilities
        for i in range(batch_size):
            prompt_tokens = self.model.tokenizer.encode(
                prompt_batch[i], add_special_tokens=True
            )
            choice_start = len(prompt_tokens) - 1

            log_prob = 0
            for j, token_id in enumerate(choice_ids):
                pos = choice_start + j
                if pos >= logits.shape[1]:
                    break

                token_logits = logits[i, pos]
                token_log_probs = F.log_softmax(token_logits, dim=-1)
                log_prob += token_log_probs[token_id]

            # Length-normalized log probability
            choice_logprobs[i, 0] = log_prob / max(1, len(choice_ids))
        
        # Return difference between second and first choice
        return choice_logprobs[:, 0]
    
    def calculate_directional_derivatives(self, dataset, concept_vectors, 
                                        concept_weights, prompt_template,
                                        completion, task_score='contrastive',
                                        demo_paths=None, demo_labels=None):
        """
        Calculate weighted directional derivatives for concepts.
        
        Args:
            dataset: Dataset to analyze
            concept_vectors: Tensor of concept direction vectors
            concept_weights: Tensor of concept importance weights
            prompt_template: Template for generating prompts
            completion: string of completion
            demo_paths: Optional demonstration image paths
            demo_labels: Optional demonstration labels
            
        Returns:
            Tuple of (weighted_sensitivities, raw_sensitivities)
        """
        if self.wrapped_layer is None:
            raise RuntimeError("No layer wrapped. Call setup_layer_hook first.")
            
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        concept_vectors = concept_vectors.cuda()
        concept_weights = concept_weights.cuda()
        
        # Process demos if provided
        stacked_demos = None
        if demo_paths is not None and demo_labels is not None:
            processed_imgs = self.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_imgs)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=1)
        
        # Build prompt
        prompt = prompt_template.build_prompt(demo_labels if demo_labels else None)
        
        # Get final token position
        encoded = self.model.tokenizer(prompt, return_tensors="pt")
        final_tok_position = encoded["input_ids"].shape[1] - 1
        
        all_raw_sensitivities = []
        all_weighted_sensitivities = []
        
        for batch in tqdm(dataloader, desc="Calculating sensitivities"):
            torch.cuda.empty_cache()
            
            image_batch = batch['image'].cuda()
            if len(image_batch.shape) == 4:
                image_batch = image_batch.unsqueeze(1).unsqueeze(2)
            
            if stacked_demos is not None:
                image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)
            
            prompt_batch = [prompt]
            
            if task_score == 'contrastive':
                # === First forward pass: malignant ===
                layer_outputs_malignant = []
                def hook_fn_malignant(module, input, output):
                    layer_outputs_malignant.append(output)
                hook_malignant = self.wrapped_layer.register_forward_hook(hook_fn_malignant)
                
                log_prob_malignant = self.compute_model_outputs(
                    image_batch, prompt_batch, " malignant"
                )
                
                activation_grad_malignant = torch.autograd.grad(
                    outputs=log_prob_malignant,
                    inputs=layer_outputs_malignant[-1][0],
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                hook_malignant.remove()
                
                # === Second forward pass: benign ===
                layer_outputs_benign = []
                def hook_fn_benign(module, input, output):
                    layer_outputs_benign.append(output)
                hook_benign = self.wrapped_layer.register_forward_hook(hook_fn_benign)
                
                log_prob_benign = self.compute_model_outputs(
                    image_batch, prompt_batch, " benign"
                )
                
                activation_grad_benign = torch.autograd.grad(
                    outputs=log_prob_benign,
                    inputs=layer_outputs_benign[-1][0],
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                hook_benign.remove()
                
                # === Combine gradients for contrastive effect ===
                # ∂(log_prob_malignant - log_prob_benign)/∂activations
                contrastive_grad = activation_grad_malignant - activation_grad_benign
                # Flatten gradient
                flattened_grad = contrastive_grad.view(contrastive_grad.size(1), -1)
                
                # Compute directional derivatives
                raw_sensitivities = torch.matmul(flattened_grad, concept_vectors.T)
                weighted_sensitivities = raw_sensitivities * concept_weights.unsqueeze(0)
                
                # Store results for final token
                all_raw_sensitivities.append(
                    raw_sensitivities.cpu().detach().numpy()[final_tok_position, :]
                )
                all_weighted_sensitivities.append(
                    weighted_sensitivities.cpu().detach().numpy()[final_tok_position, :]
                )
                
                del activation_grad_benign, activation_grad_malignant, raw_sensitivities, weighted_sensitivities
                del layer_outputs_benign[:] 
                del layer_outputs_malignant[:]
                torch.cuda.empty_cache()
            elif task_score == "malignant_prob":
                # Hook to collect layer outputs
                layer_outputs = []
                def hook_fn(module, input, output):
                    layer_outputs.append(output)
                hook = self.wrapped_layer.register_forward_hook(hook_fn)
                
                # Compute choice difference
                outputs = self.compute_model_outputs(
                    image_batch, prompt_batch, completion
                )
                
                # Compute gradient
                activation_grad = torch.autograd.grad(
                    outputs=outputs,
                    inputs=layer_outputs[-1][0],
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                hook.remove()
                
                # Flatten gradient
                flattened_grad = activation_grad.view(activation_grad.size(1), -1)
                
                # Compute directional derivatives
                raw_sensitivities = torch.matmul(flattened_grad, concept_vectors.T)
                weighted_sensitivities = raw_sensitivities * concept_weights.unsqueeze(0)
                
                # Store results for final token
                all_raw_sensitivities.append(
                    raw_sensitivities.cpu().detach().numpy()[final_tok_position, :]
                )
                all_weighted_sensitivities.append(
                    weighted_sensitivities.cpu().detach().numpy()[final_tok_position, :]
                )
                
                # At the end of each batch:
                del activation_grad, raw_sensitivities, weighted_sensitivities
                del layer_outputs[:]  # Clear the list
                torch.cuda.empty_cache()
        
        # Stack results
        all_weighted = np.vstack(all_weighted_sensitivities)
        all_raw = np.vstack(all_raw_sensitivities)
        
        return all_weighted, all_raw
    
    def image_comparison_from_paths(
        self,
        base_paths: list[str],
        alt_paths: list[str],
        prompt_template,
        completion: str,
        demo_paths = None,
        demo_labels = None,
    ):
        """
        Compare model outputs between two lists of image paths (base vs alternate).
        
        Args:
            base_paths: List of paths to base images
            alt_paths: List of paths to alternate images (same length/order)
            prompt_template: Template for generating prompts
            completion: String of completion target
            demo_paths: Optional demonstration image paths
            demo_labels: Optional demonstration labels
        
        Returns:
            Dictionary containing comparison results
        """
        if self.wrapped_layer is None:
            raise RuntimeError("No layer wrapped. Call setup_layer_hook first.")

        if len(base_paths) != len(alt_paths):
            raise ValueError("base_paths and alt_paths must be the same length.")

        # Process demo images if provided
        stacked_demos = None
        if demo_paths is not None and demo_labels is not None:
            processed_demos = self.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_demos)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=1)

        # Build prompt once
        prompt = prompt_template.build_prompt(demo_labels if demo_labels else None)
        prompt_batch = [prompt]

        results = {
            "baseline_outputs": [],
            "alternate_outputs": [],
            "diffs": [],
            "image_pairs": [],
            "completion": completion,
        }

        self.model.model.eval()

        for idx, (base_path, alt_path) in enumerate(tqdm(zip(base_paths, alt_paths), total=len(base_paths), desc="Image comparison")):
            # Load and preprocess images
            base_image = self.process_images_for_model([base_path])[0].unsqueeze(0).cuda()
            alt_image = self.process_images_for_model([alt_path])[0].unsqueeze(0).cuda()

            print(base_image.shape, alt_image.shape)
            if len(base_image.shape) == 4:
                base_image = base_image.unsqueeze(0).unsqueeze(0)
            if len(alt_image.shape) == 4:
                alt_image = alt_image.unsqueeze(0).unsqueeze(0)
            print(base_image.shape, alt_image.shape)

            # Optionally concatenate demo images
            if stacked_demos is not None:
                base_image = torch.cat([stacked_demos.cuda(), base_image], axis=1)
                alt_image = torch.cat([stacked_demos.cuda(), alt_image], axis=1)

            # Compute outputs
            with torch.no_grad():
                base_out = self.compute_model_outputs(base_image, prompt_batch, completion).item()
                alt_out = self.compute_model_outputs(alt_image, prompt_batch, completion).item()

            diff = alt_out - base_out

            results["baseline_outputs"].append(base_out)
            results["alternate_outputs"].append(alt_out)
            results["diffs"].append(diff)
            results["image_pairs"].append((base_path, alt_path))

            torch.cuda.empty_cache()

        return results


    def perturbation_analysis(self, dataset, concept_indices, prompt_template,
                             completion, perturbation_magnitude=0.2, demo_paths=None, 
                             demo_labels=None):
        """
        Test the effect of perturbing activations along concept directions.
        
        Args:
            dataset: Dataset containing test images
            concept_indices: List of concept indices to test
            prompt_template: Template for generating prompts
            completion: string of completion
            perturbation_magnitude: Magnitude of perturbation to apply
            demo_paths: Optional demonstration image paths
            demo_labels: Optional demonstration labels
            
        Returns:
            Dictionary containing perturbation results
        """
        if self.wrapped_layer is None:
            raise RuntimeError("No layer wrapped. Call setup_layer_hook first.")
        if self.concept_vectors is None:
            raise RuntimeError("No concept vectors. Call extract_concept_vectors first.")
            
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        concept_vectors = self.concept_vectors.cuda()
        
        # Process demos if provided
        stacked_demos = None
        if demo_paths is not None and demo_labels is not None:
            processed_imgs = self.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_imgs)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=1)
        
        # Build prompt
        prompt = prompt_template.build_prompt(demo_labels if demo_labels else None)
        
        # Get final token position
        encoded = self.model.tokenizer(prompt, return_tensors="pt")
        final_tok_position = encoded["input_ids"].shape[1] - 1
        
        # Results storage
        results = {
            'baseline_choice_diff': [],
            'positive_perturbations': {idx: [] for idx in concept_indices},
            'negative_perturbations': {idx: [] for idx in concept_indices},
            'image_paths': [],
            'concept_indices': concept_indices,
            'perturbation_magnitude': perturbation_magnitude,
            'completion': completion
        }
        
        self.model.model.eval()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Perturbation analysis")):
            image_batch = batch['image'].cuda()
            if len(image_batch.shape) == 4:
                image_batch = image_batch.unsqueeze(1).unsqueeze(2)
            
            if stacked_demos is not None:
                image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)
            
            prompt_batch = [prompt]
            
            # Store image path if available
            if 'path' in batch:
                results['image_paths'].append(batch['path'][0])
            else:
                results['image_paths'].append(f"image_{batch_idx}")
            
            # 1. Calculate baseline choice difference
            with torch.no_grad():
                baseline_diff = self.compute_model_outputs(
                    image_batch, prompt_batch, completion
                ).item()
            
            results['baseline_choice_diff'].append(baseline_diff)
            
            # 2. Test each concept perturbation
            for concept_idx in concept_indices:
                concept_vector = concept_vectors[concept_idx]
                
                # Test positive perturbation
                def positive_perturbation_hook(module, input, output):
                    if isinstance(output, tuple):
                        perturbed = output[0].clone()
                    else:
                        perturbed = output.clone()
                    
                    # Apply perturbation to final token
                    perturbed[0, final_tok_position, :] += perturbation_magnitude * concept_vector
                    
                    return (perturbed,) + output[1:] if isinstance(output, tuple) else perturbed
                
                hook = self.wrapped_layer.register_forward_hook(positive_perturbation_hook)
                
                try:
                    with torch.set_grad_enabled(True):
                        pos_diff = self.compute_model_outputs(
                            image_batch, prompt_batch, completion
                        ).item()
                finally:
                    hook.remove()
                
                results['positive_perturbations'][concept_idx].append(pos_diff)
                
                # Test negative perturbation
                def negative_perturbation_hook(module, input, output):
                    if isinstance(output, tuple):
                        perturbed = output[0].clone()
                    else:
                        perturbed = output.clone()
                    
                    # Apply perturbation to final token
                    perturbed[0, final_tok_position, :] -= perturbation_magnitude * concept_vector
                    
                    return (perturbed,) + output[1:] if isinstance(output, tuple) else perturbed
                
                hook = self.wrapped_layer.register_forward_hook(negative_perturbation_hook)
                
                try:
                    with torch.set_grad_enabled(True):
                        neg_diff = self.compute_model_outputs(
                            image_batch, prompt_batch, completion
                        ).item()
                finally:
                    hook.remove()
                
                results['negative_perturbations'][concept_idx].append(neg_diff)
                
            torch.cuda.empty_cache()
            
        return results
    
    def generate_with_concept_perturbation(self, dataset, concept_label, prompt_template,
                                      perturbation_magnitude=1.0, max_new_tokens=100,
                                      demo_paths=None, demo_labels=None, 
                                      temperature=1.0, do_sample=True,
                                      num_samples_per_image=3):
        """
        Generate text completions with and without concept perturbations to observe changes.

        Args:
            dataset: Dataset containing test images
            concept_label: String label of the concept to perturb (must exist in concept texts)
            prompt_template: Template for generating prompts
            perturbation_magnitude: Magnitude of perturbation to apply
            max_new_tokens: Maximum number of tokens to generate
            demo_paths: Optional demonstration image paths
            demo_labels: Optional demonstration labels
            temperature: Sampling temperature for generation
            do_sample: Whether to use sampling or greedy decoding
            num_samples_per_image: Number of completions to generate per condition per image

        Returns:
            Dictionary containing generation results and comparisons
        """
        if self.wrapped_layer is None:
            raise RuntimeError("No layer wrapped. Call setup_layer_hook first.")
        if self.concept_vectors is None:
            raise RuntimeError("No concept vectors. Call extract_concept_vectors first.")

        # Find concept index from label
        concept_idx = None
        if hasattr(self, 'concept_texts'):
            try:
                concept_idx = self.concept_texts.index(concept_label)
            except ValueError:
                raise ValueError(f"Concept '{concept_label}' not found in concept texts")
        else:
            raise RuntimeError("No concept texts found. Make sure to store concept_texts during get_embeddings.")

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        concept_vectors = self.concept_vectors.cuda()
        concept_vector = concept_vectors[concept_idx]

        # Process demos if provided
        stacked_demos = None
        if demo_paths is not None and demo_labels is not None:
            processed_imgs = self.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_imgs)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=1)

        # Build prompt
        prompt = prompt_template.build_prompt(demo_labels if demo_labels else None)

        # Get final token position
        encoded = self.model.tokenizer(prompt, return_tensors="pt")
        final_tok_position = encoded["input_ids"].shape[1] - 1

        # Results storage
        results = {
            'concept_label': concept_label,
            'concept_idx': concept_idx,
            'perturbation_magnitude': perturbation_magnitude,
            'baseline_generations': [],
            'positive_perturbation_generations': [],
            'negative_perturbation_generations': [],
            'image_paths': [],
            'prompt_used': prompt,
            'generation_params': {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'do_sample': do_sample,
                'num_samples_per_image': num_samples_per_image
            }
        }

        self.model.model.eval()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating with '{concept_label}' perturbations")):
            image_batch = batch['image'].cuda()
            if len(image_batch.shape) == 4:
                image_batch = image_batch.unsqueeze(1).unsqueeze(2)

            if stacked_demos is not None:
                image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)

            # Store image path if available
            if 'path' in batch:
                results['image_paths'].append(batch['path'][0])
            else:
                results['image_paths'].append(f"image_{batch_idx}")

            # Tokenize prompt
            prompt_encoded = self.model.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = prompt_encoded["input_ids"].cuda()
            attention_mask = prompt_encoded["attention_mask"].cuda()

            image_generations = {
                'baseline': [],
                'positive_perturbation': [],
                'negative_perturbation': []
            }

            # 1. Generate baseline completions (no perturbation)
            for sample_idx in range(num_samples_per_image):
                with torch.no_grad():
                    generated_ids = self.model.model.generate(
                        vision_x=image_batch,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False
                    )

                    # Decode only the new tokens
                    new_tokens = generated_ids.sequences[0][input_ids.shape[1]:]
                    completion = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    image_generations['baseline'].append(completion)

            # 2. Generate with positive perturbation
            def positive_perturbation_hook(module, input, output):
                if isinstance(output, tuple):
                    perturbed = output[0].clone()
                else:
                    perturbed = output.clone()

                # Apply perturbation to final token
                perturbed[0, final_tok_position, :] += perturbation_magnitude * concept_vector

                return (perturbed,) + output[1:] if isinstance(output, tuple) else perturbed

            for sample_idx in range(num_samples_per_image):
                hook = self.wrapped_layer.register_forward_hook(positive_perturbation_hook)

                try:
                    with torch.no_grad():
                        generated_ids = self.model.model.generate(
                            vision_x=image_batch,
                            lang_x=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=do_sample,
                            pad_token_id=self.model.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=False
                        )

                        # Decode only the new tokens
                        new_tokens = generated_ids.sequences[0][input_ids.shape[1]:]
                        completion = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        image_generations['positive_perturbation'].append(completion)
                finally:
                    hook.remove()

            # 3. Generate with negative perturbation
            def negative_perturbation_hook(module, input, output):
                if isinstance(output, tuple):
                    perturbed = output[0].clone()
                else:
                    perturbed = output.clone()

                # Apply perturbation to final token
                perturbed[0, final_tok_position, :] -= perturbation_magnitude * concept_vector

                return (perturbed,) + output[1:] if isinstance(output, tuple) else perturbed

            for sample_idx in range(num_samples_per_image):
                hook = self.wrapped_layer.register_forward_hook(negative_perturbation_hook)

                try:
                    with torch.no_grad():
                        generated_ids = self.model.model.generate(
                            vision_x=image_batch,
                            lang_x=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=do_sample,
                            pad_token_id=self.model.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=False
                        )

                        # Decode only the new tokens
                        new_tokens = generated_ids.sequences[0][input_ids.shape[1]:]
                        completion = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        image_generations['negative_perturbation'].append(completion)
                finally:
                    hook.remove()

            # Store results for this image
            results['baseline_generations'].append(image_generations['baseline'])
            results['positive_perturbation_generations'].append(image_generations['positive_perturbation'])
            results['negative_perturbation_generations'].append(image_generations['negative_perturbation'])

            torch.cuda.empty_cache()

        return results
    
    def analyze_generation_changes(self, generation_results, save_path=None):
        """
        Analyze and summarize the changes in generated text due to concept perturbations.

        Args:
            generation_results: Results from generate_with_concept_perturbation
            save_path: Optional path to save the analysis

        Returns:
            Dictionary containing analysis summary
        """
        import difflib
        from collections import Counter

        concept_label = generation_results['concept_label']
        analysis = {
            'concept_label': concept_label,
            'num_images': len(generation_results['baseline_generations']),
            'detailed_comparisons': [],
            'summary_statistics': {
                'avg_baseline_length': 0,
                'avg_positive_length': 0,
                'avg_negative_length': 0,
                'common_positive_changes': Counter(),
                'common_negative_changes': Counter()
            }
        }

        total_baseline_length = 0
        total_positive_length = 0
        total_negative_length = 0
        total_generations = 0

        for img_idx in range(len(generation_results['baseline_generations'])):
            image_path = generation_results['image_paths'][img_idx]
            baseline_gens = generation_results['baseline_generations'][img_idx]
            positive_gens = generation_results['positive_perturbation_generations'][img_idx]
            negative_gens = generation_results['negative_perturbation_generations'][img_idx]

            image_analysis = {
                'image_path': image_path,
                'baseline_generations': baseline_gens,
                'positive_perturbation_generations': positive_gens,
                'negative_perturbation_generations': negative_gens,
                'differences': {
                    'positive_vs_baseline': [],
                    'negative_vs_baseline': []
                }
            }

            # Compare each generation
            for i in range(len(baseline_gens)):
                baseline = baseline_gens[i]
                positive = positive_gens[i] if i < len(positive_gens) else ""
                negative = negative_gens[i] if i < len(negative_gens) else ""

                # Calculate length statistics
                total_baseline_length += len(baseline.split())
                total_positive_length += len(positive.split())
                total_negative_length += len(negative.split())
                total_generations += 1

                # Generate word-level diffs
                baseline_words = baseline.split()
                positive_words = positive.split()
                negative_words = negative.split()

                # Find differences
                pos_diff = list(difflib.unified_diff(baseline_words, positive_words, lineterm=''))
                neg_diff = list(difflib.unified_diff(baseline_words, negative_words, lineterm=''))

                image_analysis['differences']['positive_vs_baseline'].append({
                    'baseline': baseline,
                    'perturbed': positive,
                    'diff': pos_diff
                })

                image_analysis['differences']['negative_vs_baseline'].append({
                    'baseline': baseline,
                    'perturbed': negative,
                    'diff': neg_diff
                })

                # Collect word changes for summary statistics
                for line in pos_diff:
                    if line.startswith('+') and not line.startswith('+++'):
                        added_words = line[1:].split()
                        for word in added_words:
                            analysis['summary_statistics']['common_positive_changes'][word] += 1

                for line in neg_diff:
                    if line.startswith('+') and not line.startswith('+++'):
                        added_words = line[1:].split()
                        for word in added_words:
                            analysis['summary_statistics']['common_negative_changes'][word] += 1

            analysis['detailed_comparisons'].append(image_analysis)

        # Calculate average lengths
        if total_generations > 0:
            analysis['summary_statistics']['avg_baseline_length'] = total_baseline_length / total_generations
            analysis['summary_statistics']['avg_positive_length'] = total_positive_length / total_generations
            analysis['summary_statistics']['avg_negative_length'] = total_negative_length / total_generations

        # Get most common changes
        analysis['summary_statistics']['top_positive_changes'] = analysis['summary_statistics']['common_positive_changes'].most_common(10)
        analysis['summary_statistics']['top_negative_changes'] = analysis['summary_statistics']['common_negative_changes'].most_common(10)

        if save_path:
            import json
            with open(save_path, 'w') as f:
                # Convert Counter objects to dicts for JSON serialization
                analysis_copy = analysis.copy()
                analysis_copy['summary_statistics']['common_positive_changes'] = dict(analysis['summary_statistics']['common_positive_changes'])
                analysis_copy['summary_statistics']['common_negative_changes'] = dict(analysis['summary_statistics']['common_negative_changes'])
                json.dump(analysis_copy, f, indent=2)

        return analysis


    def print_generation_summary(self, generation_results, analysis_results=None, num_examples=3):
        """
        Print a readable summary of generation changes.

        Args:
            generation_results: Results from generate_with_concept_perturbation
            analysis_results: Optional results from analyze_generation_changes
            num_examples: Number of example comparisons to show
        """
        if analysis_results is None:
            analysis_results = self.analyze_generation_changes(generation_results)

        concept_label = generation_results['concept_label']
        perturbation_mag = generation_results['perturbation_magnitude']

        print(f"\n{'='*80}")
        print(f"CONCEPT PERTURBATION ANALYSIS: '{concept_label}'")
        print(f"Perturbation Magnitude: {perturbation_mag}")
        print(f"{'='*80}")

        # Summary statistics
        stats = analysis_results['summary_statistics']
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Number of images analyzed: {analysis_results['num_images']}")
        print(f"  Average baseline generation length: {stats['avg_baseline_length']:.1f} words")
        print(f"  Average positive perturbation length: {stats['avg_positive_length']:.1f} words")
        print(f"  Average negative perturbation length: {stats['avg_negative_length']:.1f} words")

        # Top changes
        if stats['top_positive_changes']:
            print(f"\nMOST COMMON WORDS IN POSITIVE PERTURBATIONS:")
            for word, count in stats['top_positive_changes'][:5]:
                print(f"  '{word}': {count} times")

        if stats['top_negative_changes']:
            print(f"\nMOST COMMON WORDS IN NEGATIVE PERTURBATIONS:")
            for word, count in stats['top_negative_changes'][:5]:
                print(f"  '{word}': {count} times")

        # Example comparisons
        print(f"\nEXAMPLE COMPARISONS:")
        print(f"{'-'*80}")

        for i in range(min(num_examples, len(analysis_results['detailed_comparisons']))):
            comparison = analysis_results['detailed_comparisons'][i]
            image_path = comparison['image_path']

            print(f"\nImage {i+1}: {image_path}")
            print(f"{'.'*40}")

            # Show first generation from each condition
            if comparison['baseline_generations']:
                baseline = comparison['baseline_generations'][0]
                positive = comparison['positive_perturbation_generations'][0] if comparison['positive_perturbation_generations'] else ""
                negative = comparison['negative_perturbation_generations'][0] if comparison['negative_perturbation_generations'] else ""

                print(f"BASELINE: {baseline}")
                print(f"POSITIVE (+{concept_label}): {positive}")
                print(f"NEGATIVE (-{concept_label}): {negative}")
                print()

        print(f"{'='*80}")