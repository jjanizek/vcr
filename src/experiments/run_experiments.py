"""
Refactored VCR experiment runner with modular configuration.

This design separates concerns into:
1. Task definitions (scoring functions)
2. Model wrappers
3. Prompt templates
4. Concept sets
5. Experiment orchestration
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable, Tuple
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpretability.vcr import ConceptAnalyzer, PromptTemplate
from interpretability.utils import CLIPEmbedder, compute_inner_products, LayerOverride
from einops import repeat
from torch.utils.data import DataLoader
from datasets.ddi import DDIDataLoader


# ============================================================================
# TASK DEFINITIONS - Define different scoring functions
# ============================================================================

class TaskScorer(ABC):
    """Abstract base class for task scoring functions."""
    
    @abstractmethod
    def compute_score(self, analyzer, image_batch, prompt_batch, **kwargs) -> float:
        """
        Compute task-specific score.
        
        Args:
            analyzer: ConceptAnalyzer instance with model
            image_batch: Preprocessed image tensors
            prompt_batch: List of prompt strings
            **kwargs: Additional arguments (e.g., labels, token IDs)
        """
        pass
    
    @abstractmethod
    def get_completion_token(self) -> str:
        """Return the completion token for directional derivatives."""
        pass


class MalignantProbScorer(TaskScorer):
    """Score = P(malignant) = exp(log P(malignant))"""
    
    def compute_score(self, analyzer, image_batch, prompt_batch, **kwargs) -> float:
        with torch.no_grad():
            # Get length-normalized log probability for " malignant"
            log_prob_malignant = analyzer.compute_model_outputs(
                image_batch, 
                prompt_batch, 
                " malignant"
            )
            # Convert to probability
            prob = torch.exp(log_prob_malignant).item()
            return prob
    
    def get_completion_token(self) -> str:
        return " malignant"


class ContrastiveScorer(TaskScorer):
    """Score = log P(malignant) - log P(benign)"""
    
    def compute_score(self, analyzer, image_batch, prompt_batch, **kwargs) -> float:
        with torch.no_grad():
            # Get log probabilities for both completions
            log_prob_malignant = analyzer.compute_model_outputs(
                image_batch, prompt_batch, " malignant"
            )
            log_prob_benign = analyzer.compute_model_outputs(
                image_batch, prompt_batch, " benign"
            )
            # Return difference (this is the log odds ratio)
            choice_diff = (log_prob_malignant - log_prob_benign).item()
            return choice_diff
    
    def get_completion_token(self) -> str:
        return " malignant"


class BCELossScorer(TaskScorer):
    """Score = -BCELoss(y_pred, y_true)
    
    Uses the log probability ratio as the logit for BCE loss.
    """
    
    def __init__(self):
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def compute_score(self, analyzer, image_batch, prompt_batch, **kwargs) -> float:
        with torch.no_grad():
            # Get log probabilities for both completions
            log_prob_malignant = analyzer.compute_model_outputs(
                image_batch, prompt_batch, " malignant"
            )
            log_prob_benign = analyzer.compute_model_outputs(
                image_batch, prompt_batch, " benign"
            )
            
            # The log odds ratio: log(P(mal)/P(ben)) = log P(mal) - log P(ben)
            # This is the natural logit to use for binary classification
            logit = log_prob_malignant - log_prob_benign
            
            # Get true label from kwargs
            label_value = kwargs.get('label')
            
            # Handle different label formats - convert to binary
            if isinstance(label_value, str):
                # Convert string label to binary
                binary_label = 1.0 if 'malignant' in label_value.lower() else 0.0
            elif isinstance(label_value, bool):
                binary_label = float(label_value)
            elif isinstance(label_value, (int, float)):
                binary_label = float(label_value)
            else:
                # Default to malignant if unclear
                print(f"Warning: unclear label format {label_value}, defaulting to 1")
                binary_label = 1.0
            
            # Ensure proper shapes for BCE
            if logit.dim() == 0:
                logit = logit.unsqueeze(0)
            
            true_label = torch.tensor([binary_label], 
                                     device=logit.device, 
                                     dtype=logit.dtype)
            
            # Compute negative BCE (higher is better)
            loss = -self.bce(logit, true_label).item()
            return loss
    
    def get_completion_token(self) -> str:
        return " malignant"


# Task registry
TASK_SCORERS = {
    'malignant_prob': MalignantProbScorer,
    'contrastive': ContrastiveScorer,
    'bce_loss': BCELossScorer,
}

# ============================================================================
# MODEL WRAPPERS - Abstract different VLMs
# ============================================================================

class VLMWrapper(ABC):
    """Abstract wrapper for Vision-Language Models."""
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass
    
    @abstractmethod
    def get_available_layers(self) -> List[str]:
        """Return list of layer names available for hooking."""
        pass
    
    @abstractmethod
    def load_model(self):
        """Load and return the model."""
        pass


class FlamingoWrapper(VLMWrapper):
    """Wrapper for OpenFlamingo models."""
    
    def __init__(self, model_variant: str = "OpenFlamingo-3B-Instruct"):
        self.model_variant = model_variant
        self.layer_mapping = {
            "OpenFlamingo-3B-Instruct": {
                "last": "model.lang_encoder.transformer.blocks.23.decoder_layer",
                "middle": "model.lang_encoder.transformer.blocks.11.decoder_layer",
                "early": "model.lang_encoder.transformer.blocks.5.decoder_layer",
            },
            "OpenFlamingo-4B": {
                "last": "model.lang_encoder.gpt_neox.layers.31.decoder_layer",
                "middle": "model.lang_encoder.gpt_neox.layers.15.decoder_layer",
                "early": "model.lang_encoder.gpt_neox.layers.7.decoder_layer",
            },
            "MedFlamingo": {
                "last": "model.lang_encoder.transformer.blocks.23.decoder_layer",
                "middle": "model.lang_encoder.transformer.blocks.11.decoder_layer",
                "early": "model.lang_encoder.transformer.blocks.5.decoder_layer",
            },
        }
    
    def get_model_name(self) -> str:
        return self.model_variant
    
    def get_available_layers(self) -> List[str]:
        return list(self.layer_mapping[self.model_variant].values())
    
    def get_layer_name(self, position: str = "last") -> str:
        """Get layer name by position (last, middle, early)."""
        return self.layer_mapping[self.model_variant][position]
    
    def load_model(self):
        from models.flamingo import FlamingoAPI
        return FlamingoAPI(self.model_variant)


class GemmaWrapper(VLMWrapper):
    """Wrapper for MedGemma models."""
    
    def __init__(self):
        self.layer_mapping = {
            "MedGemma-4B-IT": {
                "last": "model.lang_encoder.transformer.blocks.23.decoder_layer",
                "middle": "model.lang_encoder.transformer.blocks.11.decoder_layer",
                "early": "model.lang_encoder.transformer.blocks.5.decoder_layer",
            },
        }

    def get_model_name(self) -> str:
        return "MedGemma"
    
    def get_available_layers(self) -> List[str]:
        # Add appropriate layer names for Gemma
        return ["layer.23", "layer.15", "layer.7"]
    
    def load_model(self):
        from models.gemma import MedGemmaAPI
        return MedGemmaAPI()


# Model registry
VLM_MODELS = {
    'flamingo-3b-instruct': lambda: FlamingoWrapper("OpenFlamingo-3B-Instruct"),
    'flamingo-4b': lambda: FlamingoWrapper("OpenFlamingo-4B"),
    'medgemma': lambda: GemmaWrapper("MedGemma"),
    'medflamingo': lambda: FlamingoWrapper("MedFlamingo"),
}


# ============================================================================
# PROMPT TEMPLATES - Different prompting strategies
# ============================================================================

@dataclass
class PromptConfig:
    """Configuration for prompt templates."""
    base_prompt: str
    demo_template: str
    query_template: str
    use_demos: bool = False
    label_map: Dict[str, str] = field(default_factory=dict)
    
    def get_completion(self, label: str) -> str:
        """Get completion token for a given label."""
        return f" {self.label_map.get(label, label)}"


class PromptLibrary:
    """Library of prompt templates for different tasks."""
    
    @staticmethod
    def ddi_binary_classification() -> PromptConfig:
        """Binary benign/malignant classification."""
        return PromptConfig(
            base_prompt="Based on the image, this lesion is benign.<|endofchunk|>Based on the image, this lesion is malignant.<|endofchunk|>",
            demo_template="<image>Based on the image, this lesion is {label}.<|endofchunk|>",
            query_template="<image>Based on the image, this lesion is",
            use_demos=False,
            label_map={'benign': 'benign', 'malignant': 'malignant'}
        )
    
    @staticmethod
    def ddi_icl() -> PromptConfig:
        """In-context learning."""
        return PromptConfig(
            base_prompt="Based on the image, this lesion is benign.<|endofchunk|>Based on the image, this lesion is malignant.<|endofchunk|>\n\n",
            demo_template="<image>Analysis: {reasoning}\nConclusion: This lesion is {label}.<|endofchunk|>",
            query_template="<image>Analysis:",
            use_demos=True,
            label_map={'benign': 'benign', 'malignant': 'malignant'}
        )
    
    @staticmethod
    def custom(base_prompt: str, demo_template: str, query_template: str, 
               use_demos: bool = False, label_map: Dict = None) -> PromptConfig:
        """Create custom prompt configuration."""
        return PromptConfig(
            base_prompt=base_prompt,
            demo_template=demo_template,
            query_template=query_template,
            use_demos=use_demos,
            label_map=label_map or {}
        )


# ============================================================================
# CONCEPT SETS - Different concept vocabularies
# ============================================================================

@dataclass
class ConceptSetConfig:
    """Configuration for concept sets."""
    name: str
    files: List[str]
    
    def load_concepts(self) -> List[str]:
        """Load concepts from files."""
        concepts = []
        for filepath in self.files:
            with open(filepath, 'r') as f:
                concepts.extend([line.strip() for line in f if line.strip()])
        return concepts


class ConceptLibrary:
    """Library of concept sets."""
    
    @staticmethod
    def english() -> ConceptSetConfig:
        """Standard English concepts."""
        return ConceptSetConfig(
            name="english",
            files=[
                '/home/groups/roxanad/sonnet/vcr/src/concept_sets/google-10000-english-no-swears.txt',
            ]
        )
    
    @staticmethod
    def medical_only() -> ConceptSetConfig:
        """Medical concepts only."""
        return ConceptSetConfig(
            name="medical_only",
            files=['/home/groups/roxanad/sonnet/vcr/src/concept_sets/medical.txt']
        )
    
    @staticmethod
    def custom(name: str, files: List[str]) -> ConceptSetConfig:
        """Create custom concept set."""
        return ConceptSetConfig(name=name, files=files)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Master configuration for VCR experiments."""
    
    # Experiment metadata
    name: str
    results_dir: str
    
    # Model configuration
    model_key: str  # Key in VLM_MODELS registry
    layer_position: str = "last"  # or specific layer name
    
    # Task configuration
    task_scorer_key: str = "contrastive"  # Key in TASK_SCORERS registry
    
    # Prompt configuration
    prompt_config: PromptConfig = field(default_factory=PromptLibrary.ddi_binary_classification)
    
    # Concept configuration
    concept_set: ConceptSetConfig = field(default_factory=ConceptLibrary.english)
    
    # Data configuration
    metadata_path: str = ""
    data_base_dir: str = ""
    test_size: float = 0.5
    demo_size: float = 0.02
    
    # Random seeds for stability analysis
    random_seeds: List[int] = field(default_factory=lambda: list(range(25)))
    
    def get_layer_name(self, model_wrapper: VLMWrapper) -> str:
        """Get the actual layer name for hooking."""
        if hasattr(model_wrapper, 'get_layer_name'):
            return model_wrapper.get_layer_name(self.layer_position)
        return self.layer_position
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert non-serializable fields
        d['prompt_config'] = asdict(self.prompt_config)
        d['concept_set'] = asdict(self.concept_set)
        return d


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class VCRExperimentRunner:
    """Orchestrates VCR experiments with different configurations."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.clip = CLIPEmbedder()
        self.model_wrapper = VLM_MODELS[config.model_key]()
        self.task_scorer = TASK_SCORERS[config.task_scorer_key]()
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save experiment configuration."""
        with open(self.results_dir / 'experiment_config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess dataframe with label mapping."""
        df = df.copy()
        # Use the label map from prompt config
        benign_label = self.config.prompt_config.label_map.get('benign', 'benign')
        malignant_label = self.config.prompt_config.label_map.get('malignant', 'malignant')
        
        df['label'] = df['malignant'].map({
            False: benign_label,
            True: malignant_label
        })
        return df
    
    def run_single_seed(self, df_preprocessed: pd.DataFrame, 
                       random_seed: int) -> Tuple[List[str], np.ndarray]:
        """Run experiment for a single random seed."""
        
        seed_dir = self.results_dir / f'seed_{random_seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRunning seed {random_seed}...")
        
        # Initialize analyzer with current model
        model_name = self.model_wrapper.get_model_name()
        analyzer = ConceptAnalyzer(model_name, self.clip)
        
        # Get layer name and setup hook
        layer_name = self.config.get_layer_name(self.model_wrapper)
        analyzer.setup_layer_hook(layer_name, LayerOverride)
        
        # Load data
        data_loader = DDIDataLoader(
            metadata=df_preprocessed,
            base_dir=self.config.data_base_dir,
            test_size=self.config.test_size,
            demo_size=self.config.demo_size,
            random_state=random_seed
        )
        
        train_dataset, test_dataset, train_paths, demo_paths, demo_labels = \
            data_loader.get_datasets(
                analyzer.image_processor,
                use_demos=self.config.prompt_config.use_demos
            )
        
        # Build prompt template
        prompt_template = PromptTemplate(
            self.config.prompt_config.base_prompt,
            self.config.prompt_config.demo_template,
            self.config.prompt_config.query_template
        )
        
        # Get embeddings and similarity
        image_emb, text_emb, concept_texts = analyzer.get_embeddings(
            train_paths,
            self.config.concept_set.files
        )
        sim_matrix = compute_inner_products(text_emb, image_emb)
        
        # Save embeddings
        np.save(seed_dir / 'similarity_matrix.npy', sim_matrix)
        np.save(seed_dir / 'image_emb.npy', image_emb)
        np.save(seed_dir / 'text_emb.npy', text_emb)
        with open(seed_dir / 'concept_texts.json', 'w') as f:
            json.dump(concept_texts, f)
        
        # Compute task scores using the configured scorer
        print(f"Computing task scores with {self.config.task_scorer_key}...")
        choice_differences = []
        
        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        analyzer.model.model.eval()
        
        for batch in tqdm(dataloader, desc="Computing scores"):
            image_batch = batch['image'].cuda()
            if len(image_batch.shape) == 4:
                image_batch = image_batch.unsqueeze(1).unsqueeze(2)
            
            if demo_paths is not None:
                processed_imgs = analyzer.process_images_for_model(demo_paths)
                demo_batch = torch.stack(processed_imgs)
                stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=1)
                image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)
            
            prompt_batch = [prompt_template.build_prompt(demo_labels if demo_labels else None)]
            
            # Use task scorer
            score = self.task_scorer.compute_score(
                analyzer,
                image_batch,
                prompt_batch,
                label=batch.get('label', 1)
            )
            choice_differences.append(score)
        
        choice_differences = np.array(choice_differences)
        np.save(seed_dir / 'choice_differences.npy', choice_differences)
        
        # Collect activations
        print("Collecting activations...")
        activations = analyzer.collect_activations(
            train_dataset,
            prompt_template,
            demo_paths=demo_paths,
            demo_labels=demo_labels,
            batch_size=1,
            num_workers=1
        )
        
        # Train concept model
        print("Training concept model...")
        concept_results = analyzer.train_concept_model(activations, sim_matrix)
        r2_scores = concept_results['r2_scores']
        analyzer.r2_scores = r2_scores
        
        # Extract concepts and compute sensitivities
        concept_vectors = analyzer.extract_concept_vectors()
        concept_weights = analyzer.compute_concept_weights(sim_matrix)
        
        completion_token = self.task_scorer.get_completion_token()
        weighted_sens, raw_sens = analyzer.calculate_directional_derivatives(
            train_dataset,
            concept_vectors,
            concept_weights,
            prompt_template,
            completion_token,
            demo_paths=demo_paths,
            demo_labels=demo_labels
        )
        
        # Save results
        np.save(seed_dir / 'weighted_sens.npy', weighted_sens)
        np.save(seed_dir / 'raw_sens.npy', raw_sens)
        np.save(seed_dir / 'concept_weights.npy', concept_weights)
        np.save(seed_dir / 'r2_scores.npy', r2_scores)
        
        return concept_texts, weighted_sens
    
    def run_all_seeds(self):
        """Run experiments for all random seeds."""
        
        # Load and preprocess data once
        print("Loading data...")
        df = pd.read_csv(self.config.metadata_path, index_col=0)
        df_preprocessed = self.preprocess_dataframe(df)
        
        print(f"Dataset shape: {df_preprocessed.shape}")
        print(f"Label distribution: {df_preprocessed['label'].value_counts().to_dict()}")
        
        # Run for each seed
        all_sensitivities = []
        concept_texts = None
        
        for i, seed in enumerate(self.config.random_seeds):
            print(f"\n{'='*60}")
            print(f"Seed {i+1}/{len(self.config.random_seeds)}: {seed}")
            print(f"{'='*60}")
            
            seed_concept_texts, sensitivities = self.run_single_seed(
                df_preprocessed, seed
            )
            
            if concept_texts is None:
                concept_texts = seed_concept_texts
            all_sensitivities.append(sensitivities)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE!")
        print(f"Results in: {self.results_dir}")
        print(f"{'='*60}")


def main():
    """Example: Run different experiment configurations."""
    
    # Baseline experiment with contrastive scoring
    baseline_config = ExperimentConfig(
        name="baseline",
        results_dir="results/flamingo3b_contrastive_english",
        model_key="flamingo-3b-instruct",
        layer_position="last",
        task_scorer_key="contrastive",
        prompt_config=PromptLibrary.ddi_binary_classification(),
        concept_set=ConceptLibrary.english(),
        metadata_path='/scratch/users/sonnet/ddi/ddi_metadata.csv',
        data_base_dir="/scratch/users/sonnet/ddi",
        random_seeds=list(range(25))
    )
    
    runner = VCRExperimentRunner(baseline_config)
    runner.run_all_seeds()
    
    # Baseline experiment with BCE scoring
    bce_config = ExperimentConfig(
        name="baseline_bce",
        results_dir="results/flamingo3b_bce_english",
        model_key="flamingo-3b-instruct",
        layer_position="last",
        task_scorer_key="bce_loss",
        prompt_config=PromptLibrary.ddi_binary_classification(),
        concept_set=ConceptLibrary.english(),
        metadata_path='/scratch/users/sonnet/ddi/ddi_metadata.csv',
        data_base_dir="/scratch/users/sonnet/ddi",
        random_seeds=list(range(25))
    )
    
    runner2 = VCRExperimentRunner(bce_config)
    runner2.run_all_seeds()

    # Baseline experiment with malignant prob
    malignant_prob_config = ExperimentConfig(
        name="baseline_malignant_prob",
        results_dir="results/flamingo3b_malignant_prob_english",
        model_key="flamingo-3b-instruct",
        layer_position="last",
        task_scorer_key="malignant_prob",
        prompt_config=PromptLibrary.ddi_binary_classification(),
        concept_set=ConceptLibrary.english(),
        metadata_path='/scratch/users/sonnet/ddi/ddi_metadata.csv',
        data_base_dir="/scratch/users/sonnet/ddi",
        random_seeds=list(range(25))
    )
    
    runner3 = VCRExperimentRunner(malignant_prob_config)
    runner3.run_all_seeds()


if __name__ == '__main__':
    main()