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
# this first approach works locally on my lambda server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
# this second one is for sherlock
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpretability.vcr import (
    ConceptAnalyzer, 
    PromptTemplate
)
from models.flamingo import FlamingoAPI
from interpretability.utils import CLIPEmbedder, ImageDataset, compute_inner_products, LayerOverride
from sklearn.model_selection import train_test_split
from einops import repeat
from torch.utils.data import DataLoader, Dataset

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Union

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
    base_prompt: str = "Based on the image, this radiograph is normal.<|endofchunk|>Based on the image, this radiograph is abnormal.<|endofchunk|>"
    demo_template: str = "<image>Based on the image, this radiograph is {label}.<|endofchunk|>"
    query_template: str = "<image>Based on the image, this radiograph is"
    completion: str = " abnormal"
    use_demos: bool = False
    
@dataclass
class ExperimentConfig:
    """Dataclass for experiment parameters"""
    results_dir: str
    model_name: str
    layer_name: str
    metadata_path: str
    chexpert_base_dir: str
    prompt: PromptConfig = field(default_factory=PromptConfig)
    concept_files: List[str] = field(default_factory=list)
    demo_size: float = 0.02
    random_state: int = 2020
    
##
## CheXpert loading logic
## will be moved to separate datasets module in OpenSource version of software
##

class CheXpertDataLoader:
    """Simple class to handle CheXpert dataset loading and splitting."""
    
    def __init__(self, metadata: Union[str, pd.DataFrame], base_dir: str, 
                 demo_size: float = 0.02, random_state: int = 2020):
        """
        Args:
            metadata: Either path to CSV with DDI metadata or pre-loaded DataFrame
            base_dir: Base directory containing DDI images
            test_size: Fraction of data to use for test set
            demo_size: Fraction of training data to use for demos
            random_state: Random seed for reproducible splits
        """
        self.metadata = metadata
        self.base_dir = Path(base_dir)
        self.demo_size = demo_size
        self.random_state = random_state
        
        # Extract clean labels from prompt choices
        self.benign_label = "normal"
        self.malignant_label = "abnormal"
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load CSV or use provided DataFrame and validate string labels column."""
        if isinstance(self.metadata, str):
            # Load from CSV path
            self.df = pd.read_csv(self.metadata)
        elif isinstance(self.metadata, pd.DataFrame):
            # Use provided DataFrame
            self.df = self.metadata.copy()
        else:
            raise ValueError(
                f"metadata must be either a string path or pandas DataFrame, "
                f"got {type(self.metadata)}"
            )
            
        # Filter for just frontal
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        # Add patient number for proper splitting
        self.df['patient_number'] = self.df['Path'].apply(
            lambda x: x.split('train/')[1].split('/')[0]
        )
        
        # Split by patient number to avoid data leakage
        patient_numbers = self.df['patient_number'].unique()
        train_patients, test_patients = train_test_split(
            patient_numbers,
            test_size=0.5,
            random_state=self.random_state
        )
        
        # Create demo and test dataframes
        train_df = self.df[self.df['patient_number'].isin(train_patients)]
        test_df = self.df[self.df['patient_number'].isin(test_patients)]
        
        # Sample balanced positive and negative cases for both splits
        train_df_normal = train_df[train_df['No Finding'] == 1].sample(
            200, 
            random_state=self.random_state
        )
        train_df_ABnormal = train_df[train_df['No Finding'] != 1].sample(
            200, 
            random_state=self.random_state
        )
        test_df_normal = test_df[test_df['No Finding'] == 1].sample(
            200, 
            random_state=self.random_state
        )
        test_df_ABnormal = test_df[test_df['No Finding'] != 1].sample(
            200, 
            random_state=self.random_state
        )
        
        self.train_df = pd.concat([
            train_df_normal, test_df_ABnormal
        ]).reset_index(drop=True)
        self.test_df = pd.concat([
            test_df_normal, test_df_ABnormal
        ]).reset_index(drop=True)
        
        train_label_col = ['normal' if x == 1 else 'abnormal' for x in self.train_df['No Finding']]
        test_label_col = ['normal' if x == 1 else 'abnormal' for x in self.test_df['No Finding']]
        
        self.train_df['label'] = train_label_col
        self.test_df['label'] = test_label_col

    def get_datasets(self, image_processor, use_demos: bool = False):
        """
        Get train and test datasets, optionally splitting out demos.
    
        Returns:
            train_dataset: ImageDataset for training
            test_dataset: ImageDataset for testing
            train_paths: List of training image paths
            demo_paths: List of demo image paths (None if use_demos=False)
            demo_labels: List of demo labels (None if use_demos=False)
        """
        demo_paths = None
        demo_labels = None
    
        train_df_to_use = self.train_df
    
        if use_demos:
            # Split demos from training data
            train_df_to_use, demo_df = train_test_split(
                self.train_df,
                test_size=self.demo_size,
                random_state=self.random_state
            )
        
            print(f"Demo set shape: {demo_df.shape}")
        
            # Extract demo paths and labels
            demo_paths = [self.base_dir / row.Path for _, row in demo_df.iterrows()]
            demo_labels = demo_df['label'].tolist()
    
        # Create image paths
        train_paths = [self.base_dir / file for file in train_df_to_use.Path]
        test_paths = [self.base_dir / file for file in self.test_df.Path]
    
        # Create datasets
        train_dataset = ImageDataset(train_paths, image_processor)
        test_dataset = ImageDataset(test_paths, image_processor)
    
        return train_dataset, test_dataset, train_paths, demo_paths, demo_labels
    
    def get_info(self):
        """Get basic info about the dataset splits."""
        return {
            'total_samples': len(self.df),
            'train_samples': len(self.train_df),
            'test_samples': len(self.test_df),
            'label_mapping': {
                'benign': self.benign_label,
                'malignant': self.malignant_label
            }
        }
    
##
## Generate top concepts for DDI dataset
##
    
def run_single_experiment(config_dict, df_preprocessed):
    """
    Run experiment with pre-configured dataframe.
    
    Args:
        config_dict: Experiment configuration 
        df_preprocessed: DataFrame with 'label' column already set up
    """
    
    # load results dir
    results_dir = Path(config_dict['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    print(f"Loading model: {config_dict['model_name']}")
    model_name = config_dict['model_name']
    clip = CLIPEmbedder()

    # Initialize analyzer
    analyzer = ConceptAnalyzer(model_name, clip)

    # Set up layer hook
    analyzer.setup_layer_hook(config_dict['layer_name'], LayerOverride)
    
    # Load DDI data using preprocessed dataframe
    data_loader = CheXpertDataLoader(
        metadata=df_preprocessed,
        base_dir=config_dict['chexpert_base_dir'],
        demo_size=config_dict.get('demo_size', 0.02),
        random_state=config_dict.get('random_state', 2020)
    )
    
    print("Dataset info:", data_loader.get_info())
    
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
                choice_diff = analyzer.compute_model_outputs(
                    image_batch, prompt_batch, config_dict['prompt']['completion']
                ).item()
            
            choice_differences.append(choice_diff)
        
        choice_differences = np.array(choice_differences)
        np.save(results_dir / 'choice_differences.npy', choice_differences)
    
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
    
    # save layer-specific results
    layer_name_safe = config_dict['layer_name'].replace('.', '_')
    layer_dir = Path(results_dir) / layer_name_safe
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    # move this to a config arg later
    # we don't do this by default because it's like 20K by 8K by 30 layers and is like 10gb per experiment
    save_concept_vectors = True
    if save_concept_vectors:
        np.save(layer_dir / 'concept_vectors.npy', concept_vectors)
    
    np.save(layer_dir / 'weighted_sens.npy', weighted_sensitivities)
    np.save(layer_dir / 'raw_sens.npy', raw_sens)
    np.save(layer_dir / 'concept_weights.npy', concept_weights)
    np.save(layer_dir / 'r2_scores.npy', r2_scores)
    
    with open(layer_dir / 'exp_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)


def main():
    
    # Prompt configuration - edit this to change your prompt
    prompt_config = PromptConfig(
        base_prompt="Based on the image, this radiograph is normal.<|endofchunk|>Based on the image, this radiograph is abnormal.<|endofchunk|>",
        demo_template="<image>Based on the image, this radiograph is {label}.<|endofchunk|>",
        query_template="<image>Based on the image, this radiograph is", ## since the query ends w/o a space, start choices w/ a space
        completion=" abnormal",  # Note: leading spaces matter for tokenization
        use_demos=False ## change this if you want to do ICL or not
    )
    
    
    # Data splitting configuration
    data_config = {
        'demo_size': 0.02,       # Fraction of train set for demos -- maybe worth changing
        'random_state': 2020    # Random seed - change to get different splits
    }
    
    # Paths and model configuration
    base_config = {
        'results_dir': 'OpenFlamingo4B_CheXpert_ZS_residualStream', ## give a descriptive name of the results, and probably save to Oak/scratch
        'model_name': 'OpenFlamingo-4B', ## specify the model name here, if you change this, you will need to change the layer names too
        'metadata_path': '/your/path/to/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv',
        'chexpert_base_dir': '/your/path/to/chexpertchestxrays-u20210408/',
        'concept_files': ['/your/path/to/concept_sets/20k.txt'],
    }
                        
    # layer names for OpenFlamingo-4B
    i = 31
    layers_to_analyze = [f'model.lang_encoder.gpt_neox.layers.{i}.decoder_layer']
    
    # layer names for OF-3B-I
#     i = 23
#     layers_to_analyze = [f'model.lang_encoder.transformer.blocks.{i}.decoder_layer']
    
    # ===== END CONFIGURATION =====
    
    # Load and preprocess the dataframe once
    print("Loading and preprocessing CheXpert metadata...")
    df_preprocessed = pd.read_csv(base_config['metadata_path'])
    
    # Run experiments for each layer
    for layer_name in layers_to_analyze:
        print(f"\n{'='*50}")
        print(f"Running experiment for layer: {layer_name}")
        print(f"{'='*50}")
        
        # Create experiment config for this layer
        exp_config = ExperimentConfig(
            layer_name=layer_name,
            prompt=prompt_config,
            **base_config,
            **data_config  # Add data splitting config
        )
        
        exp_config_dict = asdict(exp_config)
        
        # Run the experiment
        run_single_experiment(exp_config_dict, df_preprocessed)
        
        print(f"Completed layer: {layer_name}")


if __name__ == '__main__':
    main()
