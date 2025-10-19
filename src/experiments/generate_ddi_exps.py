import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
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
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from interpretability.vcr import (
    ConceptAnalyzer, 
    PromptTemplate,
    analyze_cumulative_perturbation_results
)
from models.flamingo import FlamingoAPI
from interpretability.utils import CLIPEmbedder, ImageDataset, compute_inner_products, LayerOverride
from sklearn.model_selection import train_test_split
from einops import repeat
from torch.utils.data import DataLoader, Dataset

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

##
## load experiment data that gets re-used between layers
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
    choices: List[str] = field(default_factory=lambda: [" benign", " malignant"])
    use_demos: bool = True
    
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
    
##
## Generate top concepts for DDI dataset w/ benign vs malignant classification prompt
##
    
def run_single_experiment(config_dict):
    
    # load results dir
    results_dir = Path(config_dict['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and CLIP
    print(f"Loading model: {config_dict['model_name']}")
    model_name=config_dict['model_name']
    clip = CLIPEmbedder()

    # Create analyzer
    analyzer = ConceptAnalyzer(model_name, clip)

    # Set up layer hook
    analyzer.setup_layer_hook(config_dict['layer_name'], LayerOverride)
    
    # Load data
    df = pd.read_csv(config_dict['metadata_path'], index_col=0)
    
    # Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=0.5,
        random_state=1017
    )
    
    # Prepare demos if needed
    demo_paths = None
    demo_labels = None
    base_dir = config_dict['ddi_base_dir']
    if config_dict['prompt']['use_demos']:
        train_df, demo_df = train_test_split(
            train_df,
            test_size=0.05,
            random_state=1017
        )
        
        demo_paths = []
        demo_labels = []
        for _, row in demo_df.iterrows():
            demo_paths.append(Path(base_dir + row.DDI_file))
            demo_labels.append('malignant' if row.malignant else 'benign')
    
    # Create datasets
    train_paths = [Path(base_dir + file) for file in train_df.DDI_file]
    test_paths = [Path(base_dir + file) for file in test_df.DDI_file]
    train_dataset = ImageDataset(train_paths, analyzer.image_processor)
    test_dataset = ImageDataset(test_paths, analyzer.image_processor)
    
    # Build prompt template
    prompt_template = PromptTemplate(
        config_dict['prompt']['base_prompt'],
        config_dict['prompt']['demo_template'],
        config_dict['prompt']['query_template']
    )
    
    ##
    ## sim_matrix may be pre-computed since it will be the same
    ## across different layers
    ##
    ## likewise, the length-normalized logprobs
    ## of the target model completion (used to benchmark the
    ## 'correlation' method) will only need to be calculated once
    
    # Try to load previous data if available
    saved_data = None
    try:
        print(f"Loading sim matrix from previous run: {results_dir}")
        saved_data = load_experiment_data(results_dir)
        sim_matrix = torch.tensor(saved_data['similarity_matrix'])
        concept_texts = saved_data['concept_texts']
        choice_differences = saved_data.get('choice_differences')
    except Exception as e:
        print(f"Could not load previous sim matrix : {e}")
        print("Will recompute necessary data...")
        
    # Get or compute necessary data
    if not saved_data or 'similarity_matrix' not in saved_data:
        print("Computing CLIP similarity_matrix ...")
        
        # Get concept files
        concept_files = config_dict['concept_files']

        # Get CLIP embeddings for train probe set and concept texts
        image_emb, text_emb, concept_texts = analyzer.get_embeddings(
            train_paths, 
            concept_files
        )

        # Compute similarity matrix
        sim_matrix = compute_inner_products(text_emb, image_emb)
        
        # Save
        with open(results_dir / 'concept_texts.json', 'w') as f:
            json.dump(concept_texts, f)
            
        np.save(results_dir / 'similarity_matrix.npy', sim_matrix)
        np.save(results_dir / 'image_emb.npy', image_emb)
        np.save(results_dir / 'text_emb.npy', text_emb)

    
    # Compute choice differences if needed for correlation method
    if not saved_data or 'choice_differences' not in saved_data:
        print("Computing training set choice differences...")
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
                choice_diff = analyzer.compute_choice_difference(
                    image_batch, prompt_batch, config_dict['prompt']['choices']
                ).item()
            
            choice_differences.append(choice_diff)
        
        choice_differences = np.array(choice_differences)
        np.save(results_dir / 'choice_differences.npy', choice_differences)
    
    ##
    ## this will need to be computed every time because it's the specific
    ## layer activations
    ##
    
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
    # (how well the linear probe predicts the concept label
    #  from activations on held out test examples)
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
                    config_dict['prompt']['choices'],
                    demo_paths=demo_paths,
                    demo_labels=demo_labels
                )
    
    weighted_sensitivities = weighted_sens
    analyzer.sensitivity_scores = weighted_sens
    
    ##
    ## save layer-specific results
    layer_name_safe = config_dict['layer_name'].replace('.', '_')
    layer_dir = Path(results_dir) / layer_name_safe
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(layer_dir / 'weighted_sens.npy', weighted_sensitivities)
    np.save(layer_dir / 'raw_sens.npy', raw_sens)
    np.save(layer_dir / 'concept_weights.npy', concept_weights)
    np.save(layer_dir / 'concept_vectors.npy', concept_vectors)
    np.save(layer_dir / 'r2_scores.npy', r2_scores)
    np.save(layer_dir / 'activations.npy', activations)
    
    with open(layer_dir / 'exp_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
        
    
def main():
    
    list_of_layers = [f'model.lang_encoder.transformer.blocks.{i}.decoder_layer.attn' for i in range(0+1,23+1)]
    
    for layer_name in list_of_layers:
        
        exp_conf = ExperimentConfig(
            results_dir='OF3B_DDI_ICL_concepts',
            layer_name=layer_name,
            model_name='OpenFlamingo-3B-Instruct',
            metadata_path='/your/path/to/ddi/ddidiversedermatologyimages/ddi_metadata.csv',
            ddi_base_dir="/your/path/to/ddi/ddidiversedermatologyimages/",
            concept_files=['/your/path/to/concept_sets/20k.txt',
                           '/your/path/to/concept_sets/skincon.txt'],
        )
        
        exp_config_dict = asdict(exp_conf)
        
        run_single_experiment(exp_config_dict)
        
if __name__ == '__main__':
    main()