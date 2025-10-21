import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Union
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from interpretability.utils import ImageDataset

class DDIDataLoader:
    """Simple class to handle DDI dataset loading and splitting."""
    
    def __init__(self, metadata: Union[str, pd.DataFrame], base_dir: str, 
                 test_size: float = 0.5, demo_size: float = 0.02, random_state: int = 42):
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
        self.test_size = test_size
        self.demo_size = demo_size
        self.random_state = random_state
        
        # Extract clean labels from prompt choices
        self.benign_label = "benign"
        self.malignant_label = "malignant"
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load CSV or use provided DataFrame and validate string labels column."""
        if isinstance(self.metadata, str):
            # Load from CSV path
            self.df = pd.read_csv(self.metadata, index_col=0)
        elif isinstance(self.metadata, pd.DataFrame):
            # Use provided DataFrame
            self.df = self.metadata.copy()
        else:
            raise ValueError(
                f"metadata must be either a string path or pandas DataFrame, "
                f"got {type(self.metadata)}"
            )
        
        # Require 'label' column to exist
        if 'label' not in self.df.columns:
            raise ValueError(
                f"DataFrame must contain a 'label' column. "
                f"Expected labels: {self.benign_label}, {self.malignant_label}"
            )
        
        # Validate existing labels
        self._validate_existing_labels()
        
        # Initial train/test split
        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=self.test_size,
            random_state=self.random_state
        )
    
    def _validate_existing_labels(self):
        """Validate that existing label column matches prompt choices."""
        unique_labels = set(self.df['label'].unique())
        expected_labels = {self.benign_label, self.malignant_label}
        
        if unique_labels != expected_labels:
            raise ValueError(
                f"Existing 'label' column contains {unique_labels} "
                f"but prompt choices expect {expected_labels}. "
                f"Either update your prompt choices or remove the 'label' column "
                f"to have it auto-generated."
            )
        
        # Also validate that the mapping makes sense
        malignant_labels = set(self.df[self.df['malignant'] == True]['label'].unique())
        benign_labels = set(self.df[self.df['malignant'] == False]['label'].unique())
        
        if malignant_labels != {self.malignant_label}:
            raise ValueError(
                f"Malignant samples have labels {malignant_labels} "
                f"but expected {self.malignant_label}"
            )
            
        if benign_labels != {self.benign_label}:
            raise ValueError(
                f"Benign samples have labels {benign_labels} "
                f"but expected {self.benign_label}"
            )
        
        print(f"✓ Validated existing 'label' column matches prompt choices")
        
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
            demo_paths = [self.base_dir / row.DDI_file for _, row in demo_df.iterrows()]
            demo_labels = demo_df['label'].tolist()
    
        # Create image paths
        train_paths = [self.base_dir / file for file in train_df_to_use.DDI_file]
        test_paths = [self.base_dir / file for file in self.test_df.DDI_file]
    
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
            'train_malignant_ratio': self.train_df['malignant'].mean(),
            'test_malignant_ratio': self.test_df['malignant'].mean(),
            'label_mapping': {
                'benign': self.benign_label,
                'malignant': self.malignant_label
            }
        }