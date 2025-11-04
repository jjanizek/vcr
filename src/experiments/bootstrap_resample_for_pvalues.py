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
import pickle

# Add parent directory to path
# this second one is for sherlock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpretability.vcr import (
    ConceptAnalyzer, 
    PromptTemplate
)
from models.flamingo import FlamingoAPI
from models.gemma import MedGemmaAPI
from interpretability.utils import CLIPEmbedder, ImageDataset, compute_inner_products, LayerOverride
from sklearn.model_selection import train_test_split
from einops import repeat
from torch.utils.data import DataLoader, Dataset

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Union

from datasets.ddi import DDIDataLoader

##
## load experiment data that gets re-used between random seeds
##

def load_experiment_data(results_dir):
    """Load data from a previous experiment run."""
    
    # Load various saved data
    data = {}
    
    # Check if we have saved similarity matrix and choice differences
    if (results_dir / 'similarity_matrix.npy').exists():
        data['similarity_matrix'] = np.load(results_dir / 'similarity_matrix.npy')
    
    if (results_dir / 'choice_differences.npy').exists():
        data['choice_differences'] = np.load(results_dir / 'choice_differences.npy')
        
    if (results_dir / 'concept_texts.json').exists():
        with open(results_dir / 'concept_texts.json', 'r') as f:
            concept_texts = json.load(f)
            data['concept_texts'] = concept_texts
    
    return data

##
## Dataclasses
##
    
@dataclass
class PromptConfig:
    """Configuration for prompt templates."""
    base_prompt: str = "Based on the image, this lesion is benign.<|endofchunk|>Based on the image, this lesion is malignant.<|endofchunk|>"
    demo_template: str = "<image>Based on the image, this lesion is {label}.<|endofchunk|>"
    query_template: str = "<image>Based on the image, this lesion is"
    completion: str = " malignant"
    use_demos: bool = False
    
@dataclass
class ExperimentConfig:
    """Dataclass for experiment parameters"""
    results_dir: str
    model_name: str
    layer_name: str
    metadata_path: str
    ddi_base_dir: str
    prompt: PromptConfig = field(default_factory=PromptConfig)
    concept_files: List[str] = field(default_factory=list)
    test_size: float = 0.5
    demo_size: float = 0.02
    random_state: int = 42
    
##
## Generate top concepts for DDI dataset
##
    
def run_single_seed_experiment(config_dict, df_preprocessed, random_seed, shared_data=None):
    """
    Run experiment with pre-configured dataframe for a single random seed.
    
    Args:
        config_dict: Experiment configuration 
        df_preprocessed: DataFrame with 'label' column already set up
        random_seed: Random seed for this experiment
        shared_data: Pre-computed data that can be reused across seeds (similarity matrix, etc.)
    """
    
    # Update config with current seed
    config_dict = config_dict.copy()
    config_dict['random_state'] = random_seed
    
    # Create seed-specific subdirectory
    results_dir = Path(config_dict['results_dir'])
    seed_dir = results_dir / f'seed_{random_seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    print(f"Loading model: {config_dict['model_name']}")
    model_name = config_dict['model_name']
    clip = CLIPEmbedder()

    # Initialize analyzer
    analyzer = ConceptAnalyzer(model_name, clip)

    # Set up layer hook
    analyzer.setup_layer_hook(config_dict['layer_name'], LayerOverride)
    
    # Load DDI data using preprocessed dataframe with current seed
    data_loader = DDIDataLoader(
        metadata=df_preprocessed,
        base_dir=config_dict['ddi_base_dir'],
        test_size=config_dict.get('test_size', 0.5),
        demo_size=config_dict.get('demo_size', 0.02),
        random_state=random_seed  # Use current seed
    )
    
    print(f"Dataset info for seed {random_seed}:", data_loader.get_info())
    
    # Get datasets and demo data
    train_dataset, test_dataset, train_paths, demo_paths, demo_labels = data_loader.get_datasets(
        analyzer.image_processor, 
        use_demos=config_dict['prompt']['use_demos']
    )
    
    # Build prompt template
    prompt_template = PromptTemplate(
        config_dict['prompt']['base_prompt'],
        config_dict['prompt']['demo_template'],
        config_dict['prompt']['query_template']
    )
    
    print("Computing CLIP similarity_matrix...")

    # Get concept files
    concept_files = config_dict['concept_files']

    # Get CLIP embeddings for train probe set and concept texts
    image_emb, text_emb, concept_texts = analyzer.get_embeddings(
        train_paths, 
        concept_files
    )

    # Compute similarity matrix
    sim_matrix = compute_inner_products(text_emb, image_emb)

    # Save for this seed
    with open(seed_dir / 'concept_texts.json', 'w') as f:
        json.dump(concept_texts, f)

    np.save(seed_dir / 'similarity_matrix.npy', sim_matrix)
    np.save(seed_dir / 'image_emb.npy', image_emb)
    np.save(seed_dir / 'text_emb.npy', text_emb)

    
    # Compute choice differences for this seed
    print(f"Computing training set choice differences for seed {random_seed}...")
    choice_differences = []
    
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    analyzer.model.model.eval()
    
    for batch in tqdm(dataloader, desc="Computing choice differences"):
        image_batch = batch['image'].cuda()
        if len(image_batch.shape) == 4:
            image_batch = image_batch.unsqueeze(1).unsqueeze(2)
        
        if demo_paths is not None:
            processed_imgs = analyzer.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_imgs)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=1)
            image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)
        
        prompt_batch = [prompt_template.build_prompt(demo_labels if demo_labels else None)]
        
        with torch.no_grad():
            choice_diff = analyzer.compute_model_outputs(
                image_batch, prompt_batch, config_dict['prompt']['completion']
            ).item()
        
        choice_differences.append(choice_diff)
    
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
    concept_results = analyzer.train_concept_model(
        activations, 
        sim_matrix
    )
    
    # get r2 scores 
    r2_scores = concept_results['r2_scores']
    analyzer.r2_scores = r2_scores
    
    # Extract concept vectors
    concept_vectors = analyzer.extract_concept_vectors()
    
    # calculates empirical variance of the CLIP-defined concept similarities
    concept_weights = analyzer.compute_concept_weights(sim_matrix)
    
    # calculate directional derivatives and weight them by concept weights
    weighted_sens, raw_sens = analyzer.calculate_directional_derivatives(
                    train_dataset,
                    concept_vectors,
                    concept_weights,
                    prompt_template,
                    config_dict['prompt']['completion'],
                    demo_paths=demo_paths,
                    demo_labels=demo_labels
                )
    
    weighted_sensitivities = weighted_sens
    analyzer.sensitivity_scores = weighted_sens
    
    # Save results for this seed
    np.save(seed_dir / 'weighted_sens.npy', weighted_sensitivities)
    np.save(seed_dir / 'raw_sens.npy', raw_sens)
    np.save(seed_dir / 'concept_weights.npy', concept_weights)
    np.save(seed_dir / 'r2_scores.npy', r2_scores)
    
    with open(seed_dir / 'exp_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Return concept texts and top concept indices for stability analysis
    return concept_texts, weighted_sensitivities


def main():
    
    # ===== EXPERIMENT CONFIGURATION =====
    
    # Random seeds to test
    random_seeds = list(range(25))  # Add or remove seeds as needed
    
    # Prompt configuration - edit this to change your prompt
    prompt_config = PromptConfig(
        base_prompt="Based on the image, this lesion is benign.<|endofchunk|>Based on the image, this lesion is malignant.<|endofchunk|>",
        demo_template="<image>Based on the image, this lesion is {label}.<|endofchunk|>",
        query_template="<image>Based on the image, this lesion is", ## since the query ends w/o a space, start choices w/ a space
        completion=" malignant",  # Note: leading spaces matter for tokenization
        use_demos=False ## change this if you want to do ICL or not
    )
    
    # Data preprocessing function - edit this to change how labels are created
    def preprocess_ddi_dataframe(df):
        """
        Preprocess the DDI dataframe to add 'label' column.
        Edit this function to change how labels are mapped.
        """
        # Extract clean labels from prompt choices
        benign_label = "benign"
        malignant_label = "malignant"
        
        # Create label column - EDIT THIS MAPPING as needed
        df['label'] = df['malignant'].map({
            False: benign_label, 
            True: malignant_label
        })
        
        return df
    
    # Data splitting configuration
    data_config = {
        'test_size': 0.5,        # Fraction for test set
        'demo_size': 0.02,       # Fraction of train set for demos
    }
    
    # Paths and model configuration
    base_config = {
        'results_dir': 'MedFlamingo_DDI_ZS_LastLayer_pvalue_noswears', ## descriptive name for seed stability experiment
        'model_name': 'OpenFlamingo-3B-Instruct',
        'metadata_path': '/scratch/users/sonnet/ddi/ddi_metadata.csv',
        'ddi_base_dir': "/scratch/users/sonnet/ddi",
        'concept_files': ['/home/groups/roxanad/sonnet/vcr/src/concept_sets/google-10000-english-no-swears.txt',
                          '/home/groups/roxanad/sonnet/vcr/src/concept_sets/medical.txt'],
    }
    
    # Just the last layer for OpenFlamingo-4B
#     layer_name = 'model.lang_encoder.gpt_neox.layers.31.decoder_layer'
    
    # For OF-3B-I, use this instead:
    layer_name = 'model.lang_encoder.transformer.blocks.23.decoder_layer'
    
    # ===== END CONFIGURATION =====
    
    # Create main results directory
    results_dir = Path(base_config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess the dataframe once
    print("Loading and preprocessing DDI metadata...")
    df = pd.read_csv(base_config['metadata_path'], index_col=0)
    df_preprocessed = preprocess_ddi_dataframe(df)
    
    print(f"Preprocessed dataframe shape: {df_preprocessed.shape}")
    print(f"Label distribution: {df_preprocessed['label'].value_counts().to_dict()}")
    
    # Create experiment config
    exp_config = ExperimentConfig(
        layer_name=layer_name,
        prompt=prompt_config,
        **base_config,
        **data_config
    )
    
    exp_config_dict = asdict(exp_config)
    
    # Save overall experiment config
    with open(results_dir / 'experiment_config.json', 'w') as f:
        json.dump({
            'layer_name': layer_name,
            'random_seeds': random_seeds,
            'config': exp_config_dict
        }, f, indent=2)
    
    # Try to compute shared data once (similarity matrix) using first seed
    # This assumes the CLIP embeddings don't depend on the random seed
    shared_data = None
    try:
        print("Checking for pre-computed shared data...")
        shared_data = load_experiment_data(results_dir)
        if 'similarity_matrix' in shared_data:
            print("Found shared similarity matrix")
    except:
        print("No shared data found - will compute for first seed and reuse")
    
    # Run experiments for each random seed
    all_sensitivities = []
    concept_texts = None
    
    for i, seed in enumerate(random_seeds):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(random_seeds)} with random seed: {seed}")
        print(f"Layer: {layer_name}")
        print(f"{'='*60}")
        
        # Run the experiment for this seed
        seed_concept_texts, sensitivities = run_single_seed_experiment(
            exp_config_dict, df_preprocessed, seed, shared_data
        )
        
        # Store results
        if concept_texts is None:
            concept_texts = seed_concept_texts
        all_sensitivities.append(sensitivities)
        
        print(f"Completed seed: {seed}")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved in: {results_dir}")
    print(f"Analyzed {len(random_seeds)} random seeds for layer: {layer_name}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()