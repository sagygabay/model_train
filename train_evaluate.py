#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Learning Model Training and Evaluation Framework

This script trains and evaluates multiple deep learning models for binary
classification based on a configuration file. It logs performance metrics
and saves raw results for later visualization.
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
import timm
from PIL import Image
import logging
import sys
import csv
import time
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Setup logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# File Handler (will be configured later in main)
file_handler = None

# Global variable to track if CUDA message has been logged
_cuda_status_logged = False

# Centralized CUDA availability check
def check_cuda_availability(config=None, log_status=False):
    """
    Check if CUDA is available and should be used based on config.

    Args:
        config: Configuration dictionary that may contain CUDA settings.
        log_status: Whether to log the CUDA status (should only be True once).

    Returns:
        bool: Whether CUDA should be used.
    """
    global _cuda_status_logged

    # Check if CUDA should be used based on config
    use_cuda = True
    if config and 'model_selection' in config:
        use_cuda = config.get('model_selection', {}).get('use_cuda', True)

    cuda_available = torch.cuda.is_available() and use_cuda

    # Set CUDA seeds if available
    if cuda_available:
        torch.cuda.manual_seed_all(SEED)

    # Log status only if requested and not already logged
    if log_status and not _cuda_status_logged:
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA is available. Using GPU: {device_name}")
        else:
            if torch.cuda.is_available() and not use_cuda:
                logger.info("CUDA is available but disabled in config. Using CPU.")
            else:
                logger.info("CUDA is not available. Using CPU.")

        # Mark as logged so we don't log again
        _cuda_status_logged = True

    return cuda_available

# Global CUDA availability flag - set once at startup
CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------- #
#                            DATASET DEFINITIONS                               #
# ---------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    """
    Dataset handler for center/not_center classification.
    Returns (image, label, path).
    """
    def __init__(self, root_dir=None, transform=None, image_paths=None, labels=None):
        """
        Args:
            root_dir: Path to root directory containing labeled subdirectories ('center', 'not_center').
            transform: PyTorch transforms to apply.
            image_paths: List of image paths (alternative to root_dir).
            labels: List of labels corresponding to image_paths.
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {"not_center": 0, "center": 1} # Consistent mapping

        if image_paths is not None:
            self.image_paths = image_paths
            self.labels = labels if labels is not None else [None] * len(image_paths)
        elif root_dir is not None:
            root_dir = Path(root_dir)
            # Log a warning instead of raising an error if the directory doesn't exist
            if not root_dir.is_dir():
                 logger.warning(f"Root directory not found or is not a directory: {root_dir}. Dataset will be empty.")
            else:
                # Proceed with loading images only if the directory exists
                for class_name, label_value in self.class_to_idx.items():
                    class_path = root_dir / class_name
                    # Check if the specific class directory exists within the loop
                    if not class_path.is_dir():
                        logger.warning(f"Class directory not found: {class_path}")
                        continue # Skip to the next class_name if this one doesn't exist
                    # Iterate through images if the class directory exists
                    for img_file in class_path.iterdir():
                        # Make sure to check for hidden files or system files if necessary
                        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self.image_paths.append(str(img_file))
                            self.labels.append(label_value)
        else:
             raise ValueError("Either root_dir or image_paths must be provided.")

        if not self.image_paths:
             logger.warning(f"No images found in {root_dir if root_dir else 'provided paths'}.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error opening image {img_path}: {e}")
            # Return a dummy image and label to avoid crashing the loader
            img = Image.new('RGB', (224, 224), color = 'red')
            label = torch.tensor(-1.0, dtype=torch.float32) # Indicate error
            return img, label, img_path

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return img, label, img_path

    def get_labels(self):
        """Return all labels (used for creating balanced samplers)."""
        return [l for l in self.labels if l is not None]


class DataModule:
    """Data module that handles dataset creation, transforms, and dataloaders."""

    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data', {})
        self.model_config = config.get('model_selection', {})
        self.batch_size = self.model_config.get('batch_size', 32)
        # Use a reasonable default if not specified, adjust based on system
        self.num_workers = self.data_config.get('num_workers', min(os.cpu_count(), 4))
        # Get train/val/test directories from config
        self.train_dir = Path(self.data_config.get('train_dir', 'data/train'))
        self.val_dir = Path(self.data_config.get('val_dir', 'data/validation'))
        self.test_dir = Path(self.data_config.get('test_dir', 'data/test')) # Added test_dir

    def _get_transforms(self, input_size=224, training=False):
        """Get transforms for a specific input size."""
        # Standard ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if training:
            # Augmentations from config or defaults
            augment_config = self.data_config.get('augmentations', {})
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=augment_config.get('horizontal_flip_p', 0.5)),
                transforms.RandomVerticalFlip(p=augment_config.get('vertical_flip_p', 0.0)), # Often not used
                transforms.RandomRotation(degrees=augment_config.get('rotation_degrees', 15)),
                transforms.ColorJitter(
                    brightness=augment_config.get('brightness', 0.2),
                    contrast=augment_config.get('contrast', 0.2),
                    saturation=augment_config.get('saturation', 0.0), # Often not used
                    hue=augment_config.get('hue', 0.0) # Often not used
                ),
                transforms.ToTensor(),
                normalize
            ])
        else: # Validation/Test transforms
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize
            ])

    def _create_weighted_sampler(self, labels):
        """Create a weighted sampler for balanced training."""
        if not labels:
            return None
        try:
            class_counts = np.bincount(labels)
            # Handle cases where a class might be missing in a small dataset/split
            if len(class_counts) < 2:
                logger.warning("Only one class present in the dataset for sampling. WeightedRandomSampler might not work as expected.")
                # Fallback to standard sampling
                return None
            # Avoid division by zero if a class has zero samples (shouldn't happen with bincount on existing labels)
            class_weights = 1. / np.maximum(class_counts, 1)
            weights = [class_weights[label] for label in labels]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return sampler
        except Exception as e:
            logger.error(f"Error creating weighted sampler: {e}")
            return None


    def setup(self, input_size=224):
        """Setup train/val datasets from pre-defined directories."""
        logger.info(f"Setting up training dataset from: {self.train_dir}")
        logger.info(f"Setting up validation dataset from: {self.val_dir}")
        logger.info(f"Setting up test dataset from: {self.test_dir}") # Added test log

        # Create training dataset
        self.train_dataset = ImageDataset(
            root_dir=self.train_dir,
            transform=self._get_transforms(input_size, training=True)
        )
        if len(self.train_dataset) == 0:
            raise ValueError(f"No training images found in {self.train_dir}")

        # Create validation dataset
        self.val_dataset = ImageDataset(
            root_dir=self.val_dir,
            transform=self._get_transforms(input_size, training=False)
        )
        if len(self.val_dataset) == 0:
            raise ValueError(f"No validation images found in {self.val_dir}")

        # Create test dataset
        self.test_dataset = ImageDataset(
            root_dir=self.test_dir,
            transform=self._get_transforms(input_size, training=False)
        )
        if len(self.test_dataset) == 0:
            # Log a warning but don't raise an error, maybe testing is optional
            logger.warning(f"No test images found in {self.test_dir}. Test evaluation will be skipped.")


        logger.info(f"Loaded datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} validation, {len(self.test_dataset)} test")
        try:
            train_labels = self.train_dataset.get_labels()
            val_labels = self.val_dataset.get_labels()
            test_labels = self.test_dataset.get_labels() # Added test labels
            if train_labels:
                 logger.info(f"Train class distribution: {np.bincount(train_labels)}")
            if val_labels:
                 logger.info(f"Validation class distribution: {np.bincount(val_labels)}")
            if test_labels:
                 logger.info(f"Test class distribution: {np.bincount(test_labels)}") # Added test distribution log
        except Exception as e:
             logger.warning(f"Could not log class distributions: {e}")


    def get_train_dataloader(self, balanced=True):
        """Get training dataloader with optional class balancing."""
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Train dataset not initialized. Call setup first.")

        sampler = None
        shuffle = True
        if balanced:
            labels = self.train_dataset.get_labels()
            sampler = self._create_weighted_sampler(labels)
            if sampler:
                shuffle = False # Sampler handles shuffling

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True # Drop last incomplete batch for potentially smoother training
        )

    def get_val_dataloader(self):
        """Get validation dataloader."""
        if not hasattr(self, 'val_dataset'):
            raise ValueError("Validation dataset not initialized. Call setup first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_test_dataloader(self):
        """Get test dataloader."""
        if not hasattr(self, 'test_dataset') or len(self.test_dataset) == 0:
            # Return None if test dataset is empty or not initialized
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# ---------------------------------------------------------------------------- #
#                             MODEL DEFINITIONS                                #
# ---------------------------------------------------------------------------- #
class ClassificationModel:
    """
    Class for handling pre-trained models for classification.
    """
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.model_config = config.get('model_selection', {})
        self.pretrained = self.model_config.get('pretrained', True) # Default to True
        self.freeze_backbone = self.model_config.get('freeze_backbone', False)
        self.dropout_rate = self.model_config.get('dropout_rate', 0.2)
        self.num_classes = config.get('data', {}).get('num_classes', 1) # Default to binary (1 output + sigmoid)

        # Use the centralized CUDA check without logging status
        use_cuda = check_cuda_availability(config, log_status=False)
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # Just log which model is being initialized and on which device
        logger.info(f"Initializing model: {model_name} on {self.device.type.upper()}")

        # Initialize the model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model architecture using timm."""
        logger.info(f"Setting up {self.model_name} architecture (pretrained={self.pretrained})")

        try:
            # Use timm to create the model backbone
            self.model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=0  # Remove the original classifier head
            )

            # Get the embedding size (number of features before the classifier)
            if hasattr(self.model, 'num_features'):
                self.embedding_size = self.model.num_features
            elif hasattr(self.model, 'head') and hasattr(self.model.head, 'in_features'): # ViT models
                 self.embedding_size = self.model.head.in_features
                 self.model.head = nn.Identity() # Replace head
            elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'in_features'): # EfficientNet, ConvNeXt
                 self.embedding_size = self.model.classifier.in_features
                 self.model.classifier = nn.Identity() # Replace classifier
            elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features'): # ResNet models
                 self.embedding_size = self.model.fc.in_features
                 self.model.fc = nn.Identity() # Replace fc layer
            else:
                # Attempt to infer by passing dummy input (less reliable)
                logger.warning("Could not directly determine embedding size. Attempting inference.")
                with torch.no_grad():
                    # Determine input size from config or default
                    input_size = self.model_config.get('input_size', 224)
                    # Find the specific candidate config for this model if possible
                    candidates = self.model_config.get('candidates', [])
                    for cand in candidates:
                        if cand.get('name') == self.model_name and 'input_size' in cand:
                            input_size = cand['input_size']
                            break
                    dummy_input = torch.zeros(1, 3, input_size, input_size)
                    dummy_output = self.model(dummy_input)
                    self.embedding_size = dummy_output.shape[-1]
                logger.warning(f"Inferred embedding size: {self.embedding_size}")


        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name} using timm: {e}")
            raise ValueError(f"Model {self.model_name} could not be initialized.") from e

        logger.info(f"Model backbone initialized with embedding size: {self.embedding_size}")

        # Create a new classifier head
        # Use BCEWithLogitsLoss for binary classification (more stable than Sigmoid + BCELoss)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embedding_size, self.num_classes) # Output raw logits
        )

        # Freeze backbone layers if specified
        if self.freeze_backbone:
            logger.info("Freezing backbone layers")
            for param in self.model.parameters():
                param.requires_grad = False
            # Ensure classifier is trainable
            for param in self.classifier.parameters():
                 param.requires_grad = True

        # Move to device
        self.model.to(self.device)
        self.classifier.to(self.device)

    def get_embedding(self, x):
        """Extract embedding from the backbone model."""
        return self.model(x)

    def forward(self, x):
        """Forward pass through model and classifier."""
        embedding = self.get_embedding(x)
        output = self.classifier(embedding)
        # If binary classification, squeeze the output
        if self.num_classes == 1:
            output = output.squeeze(-1)
        return output # Return logits

    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters()) + \
                       sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return total_params, trainable_params


    def save_checkpoint(self, filepath, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_name': self.model_name,
            'embedding_size': self.embedding_size,
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            **kwargs
        }
        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Model checkpoint saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving checkpoint to {filepath}: {e}")


    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        if not Path(filepath).exists():
             logger.error(f"Checkpoint file not found: {filepath}")
             return None
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # Verify the model architecture matches
            if checkpoint.get('model_name') != self.model_name:
                logger.warning(f"Checkpoint model ({checkpoint.get('model_name')}) doesn't match current model ({self.model_name})")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

            logger.info(f"Model checkpoint loaded from {filepath}")
            return checkpoint
        except Exception as e:
             logger.error(f"Error loading checkpoint from {filepath}: {e}")
             return None


# ---------------------------------------------------------------------------- #
#                            TRAINING FUNCTIONS                                #
# ---------------------------------------------------------------------------- #
class Trainer:
    """
    Class for training and evaluating the model, including metric logging.
    """
    def __init__(self, model, config, model_output_dir):
        self.model = model
        self.config = config
        self.model_config = config.get('model_selection', {})
        self.device = model.device
        self.learning_rate = self.model_config.get('learning_rate', 1e-3)
        self.weight_decay = self.model_config.get('weight_decay', 1e-4)
        self.model_output_dir = Path(model_output_dir) # Directory for this specific model's outputs
        self.results_dir = self.model_output_dir / "results" # Subdir for raw results
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self._setup_optimizer()

        # Loss function - Use BCEWithLogitsLoss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # History tracking
        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'lr': [], 'epoch_time': []} # Added val_auc


    def _setup_optimizer(self):
        """Setup optimizer with optional differential learning rates."""
        optimizer_name = self.model_config.get('optimizer', 'Adam').lower()
        lr_backbone_factor = self.model_config.get('lr_backbone_factor', 0.1) # Factor for backbone LR

        params_to_optimize = []
        if not self.model.freeze_backbone:
             params_to_optimize.append({
                  'params': self.model.model.parameters(),
                  'lr': self.learning_rate * lr_backbone_factor
             })
        # Always optimize the classifier
        params_to_optimize.append({
             'params': self.model.classifier.parameters(),
             'lr': self.learning_rate
        })

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(params_to_optimize, weight_decay=self.weight_decay)
        elif optimizer_name == 'adamw':
             self.optimizer = optim.AdamW(params_to_optimize, weight_decay=self.weight_decay)
        elif optimizer_name == 'sgd':
             momentum = self.model_config.get('sgd_momentum', 0.9)
             self.optimizer = optim.SGD(params_to_optimize, momentum=momentum, weight_decay=self.weight_decay)
        else:
             logger.warning(f"Unsupported optimizer: {optimizer_name}. Defaulting to Adam.")
             self.optimizer = optim.Adam(params_to_optimize, weight_decay=self.weight_decay)

        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}, Base LR: {self.learning_rate}, Backbone LR Factor: {lr_backbone_factor if not self.model.freeze_backbone else 'N/A (Frozen)'}")

        # Learning Rate Scheduler
        scheduler_config = self.model_config.get('lr_scheduler', {})
        scheduler_name = scheduler_config.get('name', None)
        self.scheduler = None
        if scheduler_name:
            if scheduler_name.lower() == 'steplr':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 10),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
                logger.info(f"Using StepLR scheduler: step_size={scheduler_config.get('step_size', 10)}, gamma={scheduler_config.get('gamma', 0.1)}")
            elif scheduler_name.lower() == 'cosinelr':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.model_config.get('epochs', 10), # Use total epochs
                    eta_min=scheduler_config.get('eta_min', 0)
                )
                logger.info(f"Using CosineAnnealingLR scheduler: T_max={self.model_config.get('epochs', 10)}, eta_min={scheduler_config.get('eta_min', 0)}")
            else:
                logger.warning(f"Unsupported scheduler: {scheduler_name}. No scheduler will be used.")


    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.model.train()
        self.model.classifier.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)

        for batch in progress_bar:
            imgs, labels, _ = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device) # Keep labels as 1D

            self.optimizer.zero_grad()
            outputs = self.model.forward(imgs) # Get logits
            loss = self.criterion(outputs, labels) # BCEWithLogitsLoss expects raw logits

            loss.backward()
            # Gradient clipping (optional, add to config if needed)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            batch_loss = loss.item()
            running_loss += batch_loss * imgs.size(0)
            # Get predictions from logits
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            batch_correct = (predicted == labels).sum().item()
            total_samples += labels.size(0)
            correct_predictions += batch_correct
            batch_acc = batch_correct / imgs.size(0) if imgs.size(0) > 0 else 0

            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.1e}' # Show LR of the main group
            })

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0

        return epoch_loss, epoch_acc

    def validate(self, dataloader):
        """Validate the model."""
        self.model.model.eval()
        self.model.classifier.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels = []
        all_preds_proba = [] # Store probabilities for AUC

        progress_bar = tqdm(dataloader, desc="Validating", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for batch in progress_bar:
                imgs, labels, _ = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model.forward(imgs) # Get logits
                loss = self.criterion(outputs, labels)

                # Statistics
                batch_loss = loss.item()
                running_loss += batch_loss * imgs.size(0)
                # Get predictions and probabilities from logits
                preds_proba = torch.sigmoid(outputs)
                predicted = (preds_proba > 0.5).float()
                batch_correct = (predicted == labels).sum().item()
                total_samples += labels.size(0)
                correct_predictions += batch_correct
                batch_acc = batch_correct / imgs.size(0) if imgs.size(0) > 0 else 0

                all_labels.extend(labels.cpu().numpy())
                all_preds_proba.extend(preds_proba.cpu().numpy())

                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.4f}'
                })

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0

        # Calculate additional metrics if possible
        val_metrics = {'val_loss': epoch_loss, 'val_acc': epoch_acc}
        if len(np.unique(all_labels)) > 1: # Need at least two classes for AUC
             try:
                  val_metrics['val_auc'] = roc_auc_score(all_labels, all_preds_proba)
             except ValueError as e:
                  logger.warning(f"Could not calculate AUC: {e}")
                  val_metrics['val_auc'] = 0.0 # Or None
        else:
             val_metrics['val_auc'] = 0.0 # Or None

        return val_metrics, all_labels, (np.array(all_preds_proba) > 0.5).astype(int) # Return metrics, true labels, predicted labels


    def fit(self, train_dataloader, val_dataloader, epochs=10):
        """Train the model."""
        best_val_metric = 0.0 # Use validation accuracy by default
        metric_to_monitor = self.model_config.get('monitor_metric', 'val_acc') # e.g., 'val_acc' or 'val_auc'
        best_epoch = -1

        # Early stopping parameters
        early_stopping_config = self.model_config.get('early_stopping', {})
        early_stopping_patience = early_stopping_config.get('patience', None)
        epochs_no_improve = 0
        min_delta = early_stopping_config.get('min_delta', 0.001) # Minimum change to qualify as improvement

        epoch_progress = tqdm(range(epochs), desc=f"Training {self.model.model_name}", position=0, dynamic_ncols=True)

        for epoch in epoch_progress:
            epoch_start_time = time.time()

            # Train epoch
            train_loss, train_acc = self.train_epoch(train_dataloader)

            # Validate
            val_metrics, _, _ = self.validate(val_dataloader)
            val_loss = val_metrics['val_loss']
            val_acc = val_metrics['val_acc']
            current_val_metric = val_metrics.get(metric_to_monitor, 0.0) # Get the metric to monitor

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_metrics.get('val_auc', 0.0)) # Store AUC
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            # Update progress bar
            postfix_dict = {
                'Tr L': f'{train_loss:.3f}', 'Tr A': f'{train_acc:.3f}',
                'Vl L': f'{val_loss:.3f}', 'Vl A': f'{val_acc:.3f}',
                'LR': f'{current_lr:.1e}', 'Time': f'{epoch_time:.1f}s'
            }
            if 'val_auc' in val_metrics:
                 postfix_dict['Vl AUC'] = f"{val_metrics['val_auc']:.3f}"
            epoch_progress.set_postfix(postfix_dict)

            # Log progress
            log_msg = (f"Epoch {epoch+1}/{epochs} - "
                       f"Tr Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                       f"Vl Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                       f"LR: {current_lr:.1e} | Time: {epoch_time:.2f}s")
            if 'val_auc' in val_metrics:
                 log_msg += f" | Vl AUC: {val_metrics['val_auc']:.4f}"
            logger.info(log_msg)


            # Save best model based on the monitored metric
            if current_val_metric > best_val_metric + min_delta:
                best_val_metric = current_val_metric
                best_epoch = epoch + 1
                epochs_no_improve = 0 # Reset counter
                checkpoint_path = self.model_output_dir / "best_model.pth"
                self.model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    best_val_metric=best_val_metric,
                    metric_monitored=metric_to_monitor,
                    **val_metrics # Save all validation metrics
                )
                logger.info(f"[BEST Epoch {best_epoch}] New best model saved with {metric_to_monitor}: {best_val_metric:.4f}")
            else:
                 epochs_no_improve += 1


            # Step the scheduler
            if self.scheduler:
                # Some schedulers like ReduceLROnPlateau need metrics
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_val_metric)
                else:
                    self.scheduler.step()

            # Early stopping check
            if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement on {metric_to_monitor}.")
                break

        # Save final model
        final_checkpoint_path = self.model_output_dir / "final_model.pth"
        self.model.save_checkpoint(
            final_checkpoint_path,
            epoch=epoch, # Last completed epoch
            **val_metrics # Save last validation metrics
        )

        # Save training history to CSV
        history_df = pd.DataFrame(self.history)
        history_path = self.results_dir / "training_history.csv"
        try:
            history_df.to_csv(history_path, index=False)
            logger.info(f"Training history saved to {history_path}")
        except Exception as e:
            logger.error(f"Error saving training history to {history_path}: {e}")


        # Plot basic training history (can be replaced by visualize.py later)
        self._plot_training_history(history_df, self.model_output_dir)

        # Return path to best model for potential test evaluation
        best_model_path = self.model_output_dir / "best_model.pth"

        # Evaluate BEST model on validation set (as before, for reporting)
        logger.info("Evaluating BEST model on validation set...")
        if best_model_path.exists():
             # Load the best checkpoint specifically for this evaluation
             loaded_checkpoint = self.model.load_checkpoint(best_model_path)
             if loaded_checkpoint:
                 final_val_metrics, true_labels, pred_labels = self.validate(val_dataloader)
                 logger.info(f"Best Model Validation Metrics: {final_val_metrics}")
             else:
                 logger.error("Failed to load best model checkpoint for validation evaluation.")
                 final_val_metrics = {}
                 true_labels, pred_labels = [], []

             # Save predictions and labels for external visualization
             preds_path = self.results_dir / "best_model_validation_preds.npz"
             try:
                 np.savez(preds_path, true_labels=np.array(true_labels), pred_labels=np.array(pred_labels))
                 logger.info(f"Best model validation predictions saved to {preds_path}")
             except Exception as e:
                 logger.error(f"Error saving validation predictions: {e}")

             # Generate confusion matrix for the best model on validation set
             self._plot_confusion_matrix(true_labels, pred_labels, self.model_output_dir / "best_model_validation_confusion_matrix.png", title_suffix="(Validation Set)")
             # Generate classification report for validation set
             try:
                 report = classification_report(true_labels, pred_labels, target_names=['not_center', 'center'], output_dict=True, zero_division=0)
                 # Add AUC to the report if available
                 if 'val_auc' in final_val_metrics:
                      report['auc'] = final_val_metrics['val_auc']
                 report_df = pd.DataFrame(report).transpose()
                 report_path = self.results_dir / "best_model_validation_classification_report.csv"
             except Exception as e:
                 logger.error(f"Error generating validation classification report: {e}")
                 report_df = pd.DataFrame() # Create empty df on error
                 report_path = self.results_dir / "best_model_validation_classification_report.csv"

             try:
                 report_df.to_csv(report_path)
                 logger.info(f"Best model validation classification report saved to {report_path}")
             except Exception as e:
                 logger.error(f"Error saving classification report: {e}")

        else:
             logger.warning("Best model checkpoint not found. Skipping final evaluation.")
             final_val_metrics = {}


        return {
            'history': self.history,
            'best_val_metric': best_val_metric,
            'best_epoch': best_epoch,
            'final_val_metrics': final_val_metrics, # Metrics from best model eval on validation set
            'best_model_path': str(best_model_path) if best_model_path.exists() else None # Return path to best model
        }

    def _plot_training_history(self, history_df, output_dir):
        """Plot basic training history (Loss and Accuracy)."""
        try:
            plt.style.use('seaborn-v0_8-grid') # Use a clean style
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot losses
            ax1.plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Training Loss')
            ax1.plot(history_df['epoch'], history_df['val_loss'], 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Plot accuracies
            ax2.plot(history_df['epoch'], history_df['train_acc'], 'b-', label='Training Accuracy')
            ax2.plot(history_df['epoch'], history_df['val_acc'], 'r-', label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            # Add AUC plot if available
            if 'val_auc' in history_df.columns and history_df['val_auc'].notna().any():
                 ax3 = ax2.twinx() # instantiate a second axes that shares the same x-axis
                 color = 'tab:green'
                 ax3.set_ylabel('Validation AUC', color=color) # we already handled the x-label with ax1
                 ax3.plot(history_df['epoch'], history_df['val_auc'], color=color, linestyle='--', label='Validation AUC')
                 ax3.tick_params(axis='y', labelcolor=color)
                 ax3.legend(loc='lower right')

            fig.tight_layout() # otherwise the right y-label is slightly clipped
            output_path = output_dir / "training_history_plot.png"
            plt.savefig(output_path, dpi=300)
            plt.close(fig) # Close the figure to free memory
            logger.info(f"Basic training history plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")

    def _plot_confusion_matrix(self, true_labels, pred_labels, output_path, title_suffix=""):
        """Plots and saves the confusion matrix."""
        if not true_labels or not pred_labels:
             logger.warning(f"Cannot plot confusion matrix {title_suffix}: No labels provided.")
             return
        try:
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['not_center', 'center'],
                        yticklabels=['not_center', 'center'])
            plt.title(f'Confusion Matrix {title_suffix}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Confusion matrix {title_suffix} saved to {output_path}")
        except Exception as e:
            logger.error(f"Error plotting confusion matrix {title_suffix}: {e}")


    def evaluate(self, dataloader, checkpoint_path):
        """Evaluate a model checkpoint on a given dataloader (e.g., test set)."""
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint not found for evaluation: {checkpoint_path}")
            return None, [], []

        logger.info(f"Loading checkpoint for evaluation: {checkpoint_path}")
        loaded_checkpoint = self.model.load_checkpoint(checkpoint_path)
        if not loaded_checkpoint:
            logger.error("Failed to load checkpoint for evaluation.")
            return None, [], []

        self.model.model.eval()
        self.model.classifier.eval()

        all_labels = []
        all_preds_proba = [] # Store probabilities for AUC

        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for batch in progress_bar:
                imgs, labels, _ = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model.forward(imgs) # Get logits

                # Get predictions and probabilities from logits
                preds_proba = torch.sigmoid(outputs)
                predicted = (preds_proba > 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_preds_proba.extend(preds_proba.cpu().numpy())

                # Calculate batch accuracy for progress bar (optional)
                batch_correct = (predicted == labels).sum().item()
                batch_acc = batch_correct / imgs.size(0) if imgs.size(0) > 0 else 0
                progress_bar.set_postfix({'acc': f'{batch_acc:.4f}'})


        if not all_labels:
             logger.warning("No samples found in the evaluation dataloader.")
             return {}, [], []

        # Calculate metrics
        pred_labels = (np.array(all_preds_proba) > 0.5).astype(int)
        accuracy = np.mean(np.array(all_labels) == pred_labels)
        metrics = {'acc': accuracy}

        if len(np.unique(all_labels)) > 1: # Need at least two classes for AUC
             try:
                  metrics['auc'] = roc_auc_score(all_labels, all_preds_proba)
             except ValueError as e:
                  logger.warning(f"Could not calculate AUC during evaluation: {e}")
                  metrics['auc'] = 0.0 # Or None
        else:
             metrics['auc'] = 0.0 # Or None

        logger.info(f"Evaluation Metrics: Accuracy={metrics['acc']:.4f}, AUC={metrics.get('auc', 'N/A'):.4f}")

        return metrics, all_labels, pred_labels


# ---------------------------------------------------------------------------- #
#                                MAIN FUNCTION                                 #
# ---------------------------------------------------------------------------- #
def main():
    """Main function to run the training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Model Training and Evaluation Framework')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, help='Override output directory from config')
    parser.add_argument('--train_dir', type=str, help='Override training data directory from config')
    parser.add_argument('--val_dir', type=str, help='Override validation data directory from config')
    parser.add_argument('--test_dir', type=str, help='Override test data directory from config') # Added test_dir arg
    parser.add_argument('--clear_outputs', action='store_true', help='Clear previous model outputs within the main output directory')
    args = parser.parse_args()

    # --- Configuration Loading ---
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file: {args.config}")
        sys.exit(1)

    # Override config with command line arguments if provided
    output_dir = Path(args.output_dir or config.get('data', {}).get('output_dir', 'output'))
    # Read train/val/test dirs from args or config
    train_dir = Path(args.train_dir or config.get('data', {}).get('train_dir', 'data/train'))
    val_dir = Path(args.val_dir or config.get('data', {}).get('val_dir', 'data/validation'))
    test_dir = Path(args.test_dir or config.get('data', {}).get('test_dir', 'data/test')) # Added test_dir read
    # Update config dict with potentially overridden paths
    config['data']['output_dir'] = str(output_dir)
    config['data']['train_dir'] = str(train_dir)
    config['data']['val_dir'] = str(val_dir)
    config['data']['test_dir'] = str(test_dir) # Added test_dir update
    clear_outputs = args.clear_outputs or config.get('data', {}).get('clear_previous_outputs', False)

    # --- Setup Output Directory and Logging ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging within the main output directory
    global file_handler
    log_file = output_dir / 'training_evaluation.log'
    file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log file each run
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    logger.info("Starting Training and Evaluation Pipeline")
    logger.info(f"Using configuration: {args.config}")
    logger.info(f"Training data directory: {train_dir}")
    logger.info(f"Validation data directory: {val_dir}")
    logger.info(f"Test data directory: {test_dir}") # Added test_dir log
    logger.info(f"Output directory: {output_dir}")

    # Save config to output directory
    try:
        config_save_path = output_dir / 'config_used.json'
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration used saved to {config_save_path}")
    except Exception as e:
        logger.error(f"Error saving config file: {e}")


    # Check CUDA availability (log status once)
    check_cuda_availability(config, log_status=True)

    # --- Data Module Setup ---
    try:
        data_module = DataModule(config)
        # Setup needs input size, get from first enabled model or default
        model_candidates = config.get('model_selection', {}).get('candidates', [])
        first_enabled_model = next((m for m in model_candidates if m.get('enabled', True)), None)
        initial_input_size = 224 # Default input size
        if first_enabled_model and 'input_size' in first_enabled_model:
             initial_input_size = first_enabled_model['input_size']
        elif model_candidates: # Fallback to first candidate if none enabled
             initial_input_size = model_candidates[0].get('input_size', 224)

        data_module.setup(input_size=initial_input_size) # Setup with initial size
    except Exception as e:
        logger.error(f"Error setting up data module: {e}", exc_info=True)
        sys.exit(1)


    # --- Model Training Loop ---
    model_selection_config = config.get('model_selection', {})
    candidates = model_selection_config.get('candidates', [])
    enabled_candidates = [c for c in candidates if c.get('enabled', True)]
    epochs = model_selection_config.get('epochs', 10)

    if not enabled_candidates:
        logger.error("No enabled model candidates found in configuration.")
        sys.exit(1)

    overall_results = []

    for candidate in enabled_candidates:
        model_name = candidate['name']
        input_size = candidate.get('input_size', 224)
        logger.info(f"\n{'='*20} Training Model: {model_name} {'='*20}")

        # --- Per-Model Setup ---
        model_output_dir = output_dir / model_name
        if clear_outputs and model_output_dir.exists():
             logger.info(f"Clearing previous output for model {model_name} in {model_output_dir}")
             shutil.rmtree(model_output_dir)
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Re-setup data module if input size differs (or just update transforms)
        # For simplicity, let's assume transforms handle resizing correctly if setup once.
        # If models truly need different preprocessing beyond resize, setup needs adjustment.
        # data_module.setup(input_size=input_size) # Re-run setup if needed

        try:
            # Get dataloaders (potentially with updated transforms if setup was re-run)
            train_dataloader = data_module.get_train_dataloader(balanced=model_selection_config.get('use_weighted_sampler', True))
            val_dataloader = data_module.get_val_dataloader()

            # Initialize model
            model = ClassificationModel(model_name, config)
            total_params, trainable_params = model.count_parameters()
            logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

            # Initialize trainer
            trainer = Trainer(model, config, model_output_dir)

            # Train model
            training_results = trainer.fit(
                train_dataloader,
                val_dataloader,
                epochs=epochs
            )

            # --- Test Set Evaluation ---
            test_metrics = {}
            best_model_path = training_results.get('best_model_path')
            test_dataloader = data_module.get_test_dataloader()

            if best_model_path and test_dataloader:
                logger.info(f"\n--- Evaluating best model ({model_name}) on TEST set ---")
                # Re-initialize model and trainer to ensure clean state for evaluation
                # (or just load checkpoint into existing model if confident)
                eval_model = ClassificationModel(model_name, config)
                eval_trainer = Trainer(eval_model, config, model_output_dir) # Need trainer for evaluate method context

                test_metrics, test_true_labels, test_pred_labels = eval_trainer.evaluate(test_dataloader, best_model_path)

                if test_metrics:
                    logger.info(f"Test Set Metrics: Accuracy={test_metrics.get('acc', 'N/A'):.4f}, AUC={test_metrics.get('auc', 'N/A'):.4f}")
                    # Save test predictions
                    test_preds_path = trainer.results_dir / "best_model_test_preds.npz"
                    try:
                        np.savez(test_preds_path, true_labels=np.array(test_true_labels), pred_labels=np.array(test_pred_labels))
                        logger.info(f"Test predictions saved to {test_preds_path}")
                    except Exception as e:
                        logger.error(f"Error saving test predictions: {e}")

                    # Generate test confusion matrix
                    trainer._plot_confusion_matrix(test_true_labels, test_pred_labels, model_output_dir / "best_model_test_confusion_matrix.png", title_suffix="(Test Set)")

                    # Generate test classification report
                    try:
                        test_report = classification_report(test_true_labels, test_pred_labels, target_names=['not_center', 'center'], output_dict=True, zero_division=0)
                        if 'auc' in test_metrics:
                             test_report['auc'] = test_metrics['auc']
                        test_report_df = pd.DataFrame(test_report).transpose()
                        test_report_path = trainer.results_dir / "best_model_test_classification_report.csv"
                        test_report_df.to_csv(test_report_path)
                        logger.info(f"Test classification report saved to {test_report_path}")
                    except Exception as e:
                        logger.error(f"Error generating/saving test classification report: {e}")
                else:
                    logger.warning("Test evaluation did not return metrics.")
            elif not test_dataloader:
                 logger.info("Test dataloader is empty or unavailable. Skipping test set evaluation.")
            else:
                 logger.warning("Best model checkpoint not found. Skipping test set evaluation.")

            # Store results for this model
            model_summary = {
                'model_name': model_name,
                'input_size': input_size,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'best_epoch': training_results.get('best_epoch', -1),
                'metric_monitored': model_selection_config.get('monitor_metric', 'val_acc'),
                'best_val_metric_value': training_results.get('best_val_metric', 0.0),
                **{f"val_{k}": v for k, v in training_results.get('final_val_metrics', {}).items()}, # Add final val metrics (prefixed)
                **{f"test_{k}": v for k, v in test_metrics.items()} # Add test metrics (prefixed)
            }
            # Calculate average epoch time
            if training_results['history']['epoch_time']:
                 model_summary['avg_epoch_time_s'] = np.mean(training_results['history']['epoch_time'])

            overall_results.append(model_summary)

            # Save model-specific summary to its directory
            model_summary_path = model_output_dir / "model_summary.json"
            try:
                # Convert numpy types for JSON serialization if necessary
                serializable_summary = {}
                for k, v in model_summary.items():
                    if isinstance(v, (np.int64, np.int32)):
                        serializable_summary[k] = int(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        serializable_summary[k] = float(v)
                    else:
                        serializable_summary[k] = v
                with open(model_summary_path, 'w') as f:
                    json.dump(serializable_summary, f, indent=2)
                logger.info(f"Model summary saved to {model_summary_path}")
            except Exception as e:
                logger.error(f"Error saving model summary JSON for {model_name}: {e}")


            logger.info(f"Finished training {model_name}. Best {model_summary['metric_monitored']}: {model_summary['best_val_metric_value']:.4f} at epoch {model_summary['best_epoch']}")

        except Exception as e:
            logger.error(f"Error during training or evaluation of model {model_name}: {e}", exc_info=True)
            # Store partial results if possible
            overall_results.append({'model_name': model_name, 'status': 'failed', 'error': str(e)})
            continue # Move to the next model

    # --- Final Summary ---
    logger.info("\n" + "="*50)
    logger.info("Overall Training Summary")
    logger.info("="*50)

    if overall_results:
        results_df = pd.DataFrame(overall_results)
        # Format columns for better readability
        for col in ['total_params', 'trainable_params']:
             if col in results_df.columns:
                  results_df[col] = results_df[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
        # Add test metrics columns to formatting list
        for col in ['best_val_metric_value', 'val_loss', 'val_acc', 'val_auc', 'test_acc', 'test_auc', 'avg_epoch_time_s']:
             if col in results_df.columns:
                  # Check if column name needs prefixing for display (already prefixed in data)
                  display_col = col
                  if col.startswith('val_') and f'val_{col}' in results_df.columns:
                       display_col = f'val_{col}'
                  elif col.startswith('test_') and f'test_{col}' in results_df.columns:
                       display_col = f'test_{col}'

                  # Apply formatting only if the column exists
                  if display_col in results_df.columns:
                      results_df[display_col] = results_df[display_col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

        logger.info("\n" + results_df.to_string())

        # Save overall results to CSV
        summary_path = output_dir / "overall_training_summary.csv"
        try:
            # Save with original numeric types before formatting
            pd.DataFrame(overall_results).to_csv(summary_path, index=False, float_format='%.6f')
            logger.info(f"Overall summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Error saving overall summary: {e}")

    else:
        logger.info("No models were successfully trained.")

    logger.info("Training and Evaluation Pipeline finished.")
    logger.info(f"Outputs saved in: {output_dir}")
    logger.info("To generate publication-ready visualizations, run visualize.py with the appropriate arguments.")


if __name__ == "__main__":
    main()
