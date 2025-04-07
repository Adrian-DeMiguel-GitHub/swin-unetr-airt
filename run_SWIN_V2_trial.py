# run_trial.py
import os
import sys
import time
import gc
import random
import pprint
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import json

# Enable interactive mode for real-time plotting
plt.ion()

# Ensure stdout & stderr flush immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import optuna

import torch
import torchinfo  # Model summary
import ptflops  # Compute model complexity (FLOPs)
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Import SwinUNETR_AIRT DATASET
import importlib
from models.swin_unetr_v2 import Swin_UNETR_AIRT_v2_dataset
# Reload the module (if necessary)
importlib.reload(Swin_UNETR_AIRT_v2_dataset)
# Import the class from the reloaded module
from models.swin_unetr_v2.Swin_UNETR_AIRT_v2_dataset import SwinUNETR_AIRT_Dataset
#################################

# Import SwinUNETR_AIRT LIGHTNING MODEL
import importlib
from models.swin_unetr_v2 import Swin_UNETR_AIRT_v2_lightning_model
# Reload the module (if necessary)
importlib.reload(Swin_UNETR_AIRT_v2_lightning_model)
# Import the class from the reloaded module
from models.swin_unetr_v2.Swin_UNETR_AIRT_v2_lightning_model import SwinUNETR_AIRT_LightningModel
##################################

# UTILITIES ######################
import importlib
# Own library
import utils.data_utils
importlib.reload(utils.data_utils)
from utils.data_utils import custom_collate

# Function to apply the viridis colormap and convert to uint8
def apply_colormap_and_normalize(array, cmap, vmin, vmax):
    norm_array = (array - vmin) / (vmax - vmin)  # Normalize to [0, 1]
    norm_array = np.clip(norm_array, 0, 1)  # Clip values to avoid overflow
    colormap = plt.get_cmap(cmap)
    rgba_img = colormap(norm_array)  # Apply colormap, returns RGBA
    rgb_img = (rgba_img[:, :, :3] * 255).astype("uint8")  # Convert to RGB
    return rgb_img
################################

# Retrieve variables from command-line arguments
study_name = sys.argv[1]
study_storage_path = sys.argv[2]
DATA_DIR = sys.argv[3]
tuning_dir = sys.argv[4]
NUM_WORKERS = int(sys.argv[5])
###############################

# Read JSON files
with open(os.path.join(tuning_dir, "temporary_files", "train_test_files.json"), 'r') as f:
    train_test_files = json.load(f)

with open(os.path.join(tuning_dir, "temporary_files", "validation_files.json"), 'r') as f:
    validation_files = json.load(f)
###############################

# Load or create the study
sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(study_name=study_name, direction="minimize", storage=study_storage_path, sampler=sampler, load_if_exists=True)
###############################

def objective(trial):

    # Initialize user-defined trial attributes
    trial.set_user_attr("model_total_params", 0)
    trial.set_user_attr("model_trainable_params", 0)
    trial.set_user_attr("model_non_trainable_params", 0)
    trial.set_user_attr("model_gflops_inference", 0) 
    trial.set_user_attr("initial_allocated_cuda_memory", 0)
    trial.set_user_attr("initial_reserved_cuda_memory", 0)
    trial.set_user_attr("final_allocated_cuda_memory", 0)
    trial.set_user_attr("final_reserved_cuda_memory", 0)

    try:

        ########## TRAINING/INFERENCE HYPERPARAMETERS #########
    
        lr_optimizer = trial.suggest_float("lr_optimizer", 1e-5, 1e-3, log=True)  # Log scale for learning rates
        weight_decay_optimizer = trial.suggest_float("weight_decay_optimizer", 1e-6, 1e-2, log=True) # Log scale for weight decay
        warmup_epochs_lr_scheduler = trial.suggest_int("warmup_epochs_lr_scheduler", 3, 10)  # Integer range
        max_training_epochs = trial.suggest_categorical("max_training_epochs", [10000])
        patience_epochs_early_stopping = trial.suggest_int("patience_epochs_early_stopping", 7, 20)  # Integer range
        batch_size_training = trial.suggest_int("batch_size_training", 1, 4)
        batch_size_inference = trial.suggest_categorical("batch_size_inference", [1]) # It doesn't matter the batch size. It always processes the patches sequentially inside the batch.
        num_batches_grad_accumulation = trial.suggest_int("num_batches_grad_accumulation", 2, 6, step=2)
        post_processing_strategy = trial.suggest_categorical("post_processing_strategy", ["fill_first","weighted_average"]) 
        pre_processing_strategy = trial.suggest_categorical("pre_processing_strategy", ["uniform_gaussian_smoothed_equidistant_subsampling","hotspot_centered_gaussian_smoothed_equidistant_subsampling"])
        
        ########## DATA PREPROCESSING HYPERPARAMETERS #########
        
        overlap_training = trial.suggest_float("overlap_training", 0, 0.3, step = 0.15)
        overlap_training = (overlap_training, overlap_training)
        overlap_inference = trial.suggest_categorical("overlap_inference", ["(0,0)"])

        ########################################################################
        
        training_config={
            "lr_optimizer": lr_optimizer,
            "weight_decay_optimizer": weight_decay_optimizer,
            "warmup_epochs_lr_scheduler": warmup_epochs_lr_scheduler,
            "max_training_epochs": max_training_epochs,
            "patience_epochs_early_stopping": patience_epochs_early_stopping,
            "batch_size_training": batch_size_training,
            "batch_size_inference": batch_size_inference,
            "num_batches_grad_accumulation": num_batches_grad_accumulation,
            "overlap_training": overlap_training,
            "overlap_inference": eval(overlap_inference),
            "post_processing_strategy": post_processing_strategy,
            "pre_processing_strategy": pre_processing_strategy
        }

        ########## MODEL HYPERPARAMETERS #########

        # Use SWIN Version 2
        use_SWIN_v2 = trial.suggest_categorical("use_SWIN_v2", [False]) # It adds a residual convolution block at the beginning of each swin layer. (with instance normalization)
        
        # Architecture
        model_spatial_input_dim = trial.suggest_int("model_spatial_input_dims", 64, 192, step=32)
        exponent_model_temporal_input_dim = trial.suggest_int("exponent_model_temporal_input_dim", 0, 3)
        model_temporal_input_dim = 64*(2**exponent_model_temporal_input_dim)
        model_input_dimensions = (model_spatial_input_dim, model_spatial_input_dim) + (model_temporal_input_dim,)
        exponent_initial_feature_embedding_size = trial.suggest_int("exponent_initial_feature_embedding_size", 0, 3)
        initial_feature_embedding_size = 12*(2**exponent_initial_feature_embedding_size)
        patch_embedding_size = trial.suggest_categorical("patch_embedding_size", [2])
        num_swin_transformer_blocks_in_layers = trial.suggest_categorical("num_swin_transformer_blocks_in_layers", [2])
        multilayer_perceptron_expansion_ratio_transformer_block=trial.suggest_categorical("mlp_ratio", [4.0]) # Defines the expansion of the MLP's hidden dimension relative to its input dimension

        # Attention
        initial_attention_head = trial.suggest_int("initial_attention_head", 2, 3, step=1)
        attention_heads = (initial_attention_head, initial_attention_head*2, initial_attention_head*4, initial_attention_head*8)
        attention_window_size = trial.suggest_int("attention_window_size", 7, 8, step=1)
        attention_qkv_bias_projections = trial.suggest_categorical("attention_qkv_projections_bias", [True])
        attention_weights_drop_rate =  trial.suggest_categorical("attention_weights_drop_rate", [0]) #[0, 0.1, 0.2]

        # Normalization
        layer_normalization_after_each_layer_in_SWIN_Transformer = trial.suggest_categorical("layer_normalization_after_each_layer_in_SWIN_Transformer", [True])
        type_normalization_in_UNET_block = trial.suggest_categorical("type_normalization_in_UNET_block", ["instance"])
        normalization_after_patch_embedding = trial.suggest_categorical("normalization_after_patch_embedding", [False])

        # Regularization
        transformer_block_drop_rate = trial.suggest_categorical("transformer_block_drop_rate", [0]) #[0, 0.1, 0.3]
        transformer_block_residual_block_dropout_path_rate = trial.suggest_categorical("transformer_block_residual_block_dropout_path_rate", [0]) #[0, 0.1, 0.2]

        # Architecture Configuration
        architecture_config={
            "use_SWIN_v2": use_SWIN_v2,
            "model_input_dimensions": model_input_dimensions,
            "model_input_channels": 1,
            "model_output_channels": 2, # 2 output channels = 2 output classes (i.e. defective/non-defective)
            "initial_feature_embedding_size": initial_feature_embedding_size,
            "patch_embedding_size": (patch_embedding_size, patch_embedding_size, patch_embedding_size),
            "num_swin_transformer_blocks_in_layers": (num_swin_transformer_blocks_in_layers, num_swin_transformer_blocks_in_layers, num_swin_transformer_blocks_in_layers, num_swin_transformer_blocks_in_layers),
            "mlp_ratio": multilayer_perceptron_expansion_ratio_transformer_block,
        }
        
        # Attention Configuration
        attention_config={
            "heads": attention_heads,
            "window_size": (attention_window_size, attention_window_size, attention_window_size),
            "qkv_bias": attention_qkv_bias_projections,
            "drop_rate": attention_weights_drop_rate,
        }
        
        # Normalization Configuration
        normalization_config={
            "use_norm_in_swinViT_after_layer": layer_normalization_after_each_layer_in_SWIN_Transformer,
            "patch_norm_in_swinViT": normalization_after_patch_embedding,
            "unet_block_norm_type": type_normalization_in_UNET_block
        }
        
        # Regularization Configuration
        regularization_config={
            "transformer_block_drop_rate": transformer_block_drop_rate,
            "transformer_block_residual_block_dropout_path_rate": transformer_block_residual_block_dropout_path_rate,
        }

        # Combine all configs into a single dictionary
        trial_config = {
            "TRAINING_CONFIG": training_config,
            "ARCHITECTURE_CONFIG": architecture_config,
            "ATTENTION_CONFIG": attention_config,
            "NORMALIZATION_CONFIG": normalization_config,
            "REGULARIZATION_CONFIG": regularization_config,
        }

        print()
        print(f"{'=' * 50}")
        print(f"{'=' * 16} RUNNING TRIAL: {trial.number} {'=' * 16}")
        print(f"{'=' * 16} Time: {datetime.now()} {'=' * 16}")
        print(f"{'=' * 50}")
        print()
        
        # Pretty-print the configurations
        pprint.pprint(trial_config, sort_dicts=False, width=100)

        print()

        # Log CUDA memory at the start of the trial
        if torch.cuda.is_available():
            trial.set_user_attr("initial_allocated_cuda_memory", torch.cuda.memory_allocated() / 1e6)
            trial.set_user_attr("initial_reserved_cuda_memory", torch.cuda.memory_reserved() / 1e6)
            trial.set_user_attr("final_allocated_cuda_memory", 0)
            trial.set_user_attr("final_reserved_cuda_memory", 0)
        
        ########## DATA PREPARATION #########

        # Creation Datasets 
        train_dataset = SwinUNETR_AIRT_Dataset(
            is_inference_mode = False,
            augmentation = True,
            metadata_dict_with_files_selected=train_test_files.copy(),
            data_dir=DATA_DIR,
            model_input_dims=model_input_dimensions,
            overlap = overlap_training,
            preprocessing_strategy = pre_processing_strategy
        )

        val_dataset = SwinUNETR_AIRT_Dataset(
            is_inference_mode = True,
            augmentation = True,
            metadata_dict_with_files_selected=validation_files.copy(),
            data_dir=DATA_DIR,
            model_input_dims=model_input_dimensions,
            overlap = eval(overlap_inference),
            preprocessing_strategy = pre_processing_strategy
        )

        # Configure DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_training, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_inference, collate_fn=custom_collate, num_workers=NUM_WORKERS)
        
        ########## MODEL INSTANTIATION #########
        
        model = SwinUNETR_AIRT_LightningModel(
            # Training Configuration
            training_config=training_config,            
            # Architecture Configuration
            architecture_config=architecture_config,            
            # Attention Configuration
            attention_config=attention_config,
            # Normalization Configuration
            normalization_config=normalization_config,         
            # Regularization Configuration
            regularization_config=regularization_config
        )

        ########## TRAINER INSTANTIATION #########

        # Early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=patience_epochs_early_stopping,
            mode='min'
        )

        # Define logger
        logger = CSVLogger(
            save_dir=tuning_dir,
            name="",
            version="")
        
        trainer = Trainer(
            max_epochs=max_training_epochs,
            callbacks=[early_stopping_callback],
            precision="16-mixed",  # Enable FP16 mixed precision
            accumulate_grad_batches=num_batches_grad_accumulation,
            enable_checkpointing=False,
            log_every_n_steps=1,
            logger=logger,
            enable_progress_bar=False,
            # Add timeout for this trial
            # max_time="00:00:02:00"  # Limit TIME (DD:HH:MM:SS format)
        )

        ########## TRAINING #########

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        input_size = (batch_size_inference, 1, model_input_dimensions[0], model_input_dimensions[1],model_input_dimensions[2])

        ### Get Total, Trainable, Non-Trainable Parameters
        model_summary = torchinfo.summary(model.model, input_size=input_size, verbose=0)

        # Store parameters in variables
        total_params = model_summary.total_params
        trainable_params = model_summary.trainable_params
        non_trainable_params = total_params - trainable_params

        ### Get FLOPs (Floating Point Operations)
        # Input size for ptflops is without batch size: (channels, height, width)
        macs, params = ptflops.get_model_complexity_info(model.model, input_size[1:], as_strings=False, verbose=False)
        # FLOPs are twice the MACs (Multiply-Accumulate Operations)
        if macs is None:
            raise torch.cuda.OutOfMemoryError()
            
        flops = 2 * macs

        trial.set_user_attr("model_total_params", total_params)
        trial.set_user_attr("model_trainable_params", trainable_params)
        trial.set_user_attr("model_non_trainable_params", non_trainable_params)
        trial.set_user_attr("model_gflops_inference", flops / 1e9) # Convert FLOPs to GFLOPs

        print()
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Non-Trainable Parameters: {non_trainable_params}")
        print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")

        # Retrieve the best validation loss
        metrics_file = os.path.join(tuning_dir, 'metrics.csv')
        metrics_df = pd.read_csv(metrics_file)
        
        # Initialize variables to make them accessible outside the if scope
        lowest_val_loss = None
        lowest_epoch = None

        # Ensure val_loss column exists and drop NaN values
        if 'val_loss_epoch' in metrics_df.columns:
            val_loss_series = metrics_df['val_loss_epoch'].dropna()
            # Find the lowest val_loss
            lowest_val_loss = val_loss_series.min()
            # Find the epoch corresponding to the lowest val_loss
            lowest_val_loss_row = metrics_df.loc[val_loss_series.idxmin()]
            lowest_epoch = lowest_val_loss_row['epoch']
        
            print(f"Lowest val_loss: {lowest_val_loss} at epoch {lowest_epoch}")
        else:
            print(f"val_loss column not found in {metrics_file}")

        ######### PLOTTING ##########

        ##### PLOT BEST VAL RESULTS #####

        print()
        print(f"{'=' * 10} PLOTTING BEST EPOCH VALIDATION RESULTS {'=' * 10}")
        print()

        # Control the maximum number of samples to show
        num_samples_to_show = 9
        
        # Adjust the number of samples if fewer samples are available
        available_samples = list(zip(
            model.best_epoch_val_sample_ids,
            model.best_epoch_val_ground_truths,
            model.best_epoch_val_predictions
        ))
        
        if len(available_samples) < num_samples_to_show:
            print(f"Number of available samples ({len(available_samples)}) is less than requested ({num_samples_to_show}). Plotting all available samples.")
            num_samples_to_show = len(available_samples)
        
        # Randomly sample the data if more samples are available than requested
        plot_samples = random.sample(available_samples, num_samples_to_show)
        
        # Calculate the number of rows needed (3 pairs per row)
        pairs_per_row = 3
        num_rows = (num_samples_to_show + pairs_per_row - 1) // pairs_per_row  # Ceiling division

        # Find unique classes dynamically
        all_classes = set()
        for gt, pred in zip(model.best_epoch_val_ground_truths, model.best_epoch_val_predictions):
            all_classes.update(torch.unique(gt).tolist())  # Add classes from ground truth
            all_classes.update(torch.unique(pred).tolist())  # Add classes from predictions
        
        # Sort the classes to ensure order
        all_classes = sorted(all_classes)
        
        # Define class labels dynamically (for simplicity, use numeric labels for now)
        class_labels = {cls: f"Class {cls}" for cls in all_classes}
        num_classes = len(class_labels)
        
        # Create a discrete colormap with exactly `num_classes` colors
        colormap = plt.cm.get_cmap("viridis", num_classes)
        
        # Create legend patches using discrete colors from the colormap
        legend_patches = [
            mpatches.Patch(color=colormap(i), label=f"{i}: {label}")
            for i, label in class_labels.items()
        ]
        # Create subplots
        fig, axes = plt.subplots(num_rows, pairs_per_row * 2, figsize=(15, 5 * num_rows))
        
        # If there's only one row, ensure axes is 2D
        if num_rows == 1:
            axes = [axes]
        
        # Flatten axes for easier indexing
        axes = [ax for row_axes in axes for ax in (row_axes if isinstance(row_axes, (list, np.ndarray)) else [row_axes])]
        
        # Plot ground truth and predictions
        for idx, (sample_id, ground_truth, prediction) in enumerate(plot_samples):
            # Calculate column index (each pair takes two columns)
            col_idx = idx * 2
        
            # Ground truth
            axes[col_idx].imshow(ground_truth.cpu().numpy(), cmap="viridis", interpolation="none",
                                 vmin=min(all_classes), vmax=max(all_classes))
            axes[col_idx].set_title(f"Sample {sample_id}\n Ground Truth \n [{ground_truth.shape[0]} x {ground_truth.shape[1]}]")
            axes[col_idx].axis("off")
        
            # Prediction
            axes[col_idx + 1].imshow(prediction.cpu().numpy(), cmap="viridis", interpolation="none",
                                     vmin=min(all_classes), vmax=max(all_classes))
            axes[col_idx + 1].set_title(f"Sample {sample_id}\n Prediction \n [{prediction.shape[0]} x {prediction.shape[1]}]")
            axes[col_idx + 1].axis("off")
        
        # Hide any unused subplots
        for unused_ax in axes[len(plot_samples) * 2:]:
            unused_ax.axis("off")
        
        # Add legend
        fig.legend(
            handles=legend_patches,
            loc="upper center",  # Places the legend below the plot
            ncol=len(all_classes),
            bbox_to_anchor=(0.5, 0.9),  # Adjusts position below the plot
            fontsize=12
        )
        
        # Add title
        fig.suptitle(
            f"Trial {trial.number}\nGround Truth vs Prediction\n"
            f"Lowest Val Loss: {lowest_val_loss:.4f} - Lowest Epoch: {lowest_epoch}",
            fontsize=16, y=1.10
        )
        
        
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        # plt.show()
        # plt.pause(0.1)
        plt.show(block=True)
        
        plt.close(fig)

        ##### PLOT TRAINING & VALIDATION LOSSES EVOLUTION ##### 

        print()
        print(f"{'=' * 10} PLOTTING TRAINING & VALIDATION LOSSES EVOLUTION {'=' * 10}")
        print()
        
        # Path to the latest metrics file
        metrics_file = os.path.join(tuning_dir, 'metrics.csv')
        metrics_df = pd.read_csv(metrics_file)
        
        # Handle train_loss and val_loss separately
        train_loss = metrics_df[['epoch', 'train_loss_epoch']].dropna().rename(columns={'train_loss_epoch': 'loss'})
        train_loss['type'] = 'Training Loss'
        
        val_loss = metrics_df[['epoch', 'val_loss_epoch']].dropna().rename(columns={'val_loss_epoch': 'loss'})
        val_loss['type'] = 'Validation Loss'
        
        # Combine the two datasets for seaborn
        plot_data = pd.concat([train_loss, val_loss], axis=0)
        
        # Plot using seaborn
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=plot_data, x='epoch', y='loss', hue='type', marker="o", style="type", dashes=False)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.grid(True)
        plt.legend(title="Loss Type")
        # plt.show()
        # plt.pause(0.1)
        plt.show(block=True)

        # Directory to save individual SVG images
        best_trial_dir = os.path.join(tuning_dir, "best_trial")
        os.makedirs(best_trial_dir, exist_ok=True)  # Create the folder if it doesn't exist

        if hasattr(trial, "study"):
            completed_trials = [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        else: # It is initial fixed trial
            completed_trials = []

        if len(completed_trials) == 0:
             best_trial_value_study = float('inf')
        else:
             best_trial_value_study = trial.study.best_trial.value

        if len(completed_trials) == 0 or best_trial_value_study > lowest_val_loss:

            ##### SAVING PLOT BEST TRIAL TRAINING/VALIDATION LOSS EVOLUTION ##### 

            # Path to the latest metrics file
            metrics_file = os.path.join(tuning_dir, 'metrics.csv')
            metrics_df = pd.read_csv(metrics_file)
            
            # Handle train_loss and val_loss separately
            train_loss = metrics_df[['epoch', 'train_loss_epoch']].dropna().rename(columns={'train_loss_epoch': 'loss'})
            train_loss['type'] = 'Training Loss'
            
            val_loss = metrics_df[['epoch', 'val_loss_epoch']].dropna().rename(columns={'val_loss_epoch': 'loss'})
            val_loss['type'] = 'Validation Loss'
            
            # Combine the two datasets for seaborn
            plot_data = pd.concat([train_loss, val_loss], axis=0)
            
            # Plot using seaborn
            fig = plt.figure(figsize=(12, 8))
            sns.lineplot(data=plot_data, x='epoch', y='loss', hue='type', marker="o", style="type", dashes=False)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Over Epochs")
            plt.grid(True)
            plt.legend(title="Loss Type")

            # Save as SVG
            plt.tight_layout()
            result_plot_path = os.path.join(best_trial_dir, f"train_val_loss_evolution.svg")
            fig.savefig(result_plot_path, format="svg", bbox_inches="tight")
            print(f"Best Trial Training/Vlaidation Loss Evolution plot saved at: {result_plot_path}")
            plt.close(fig)
            
            ##### SAVING PLOT BEST TRIAL VALIDATION RESULTS #####            
            
            # Control the maximum number of samples to show
            num_samples_to_show = 9
            
            # Adjust the number of samples if fewer samples are available
            available_samples = list(zip(
                model.best_epoch_val_sample_ids,
                model.best_epoch_val_ground_truths,
                model.best_epoch_val_predictions
            ))
            
            if len(available_samples) < num_samples_to_show:
                print(f"Number of available samples ({len(available_samples)}) is less than requested ({num_samples_to_show}). Plotting all available samples.")
                num_samples_to_show = len(available_samples)
            
            # Randomly sample the data if more samples are available than requested
            plot_samples = random.sample(available_samples, num_samples_to_show)
            
            # Calculate the number of rows needed (3 pairs per row)
            pairs_per_row = 3
            num_rows = (num_samples_to_show + pairs_per_row - 1) // pairs_per_row  # Ceiling division

            # Find unique classes dynamically
            all_classes = set()
            for gt, pred in zip(model.best_epoch_val_ground_truths, model.best_epoch_val_predictions):
                all_classes.update(torch.unique(gt).tolist())  # Add classes from ground truth
                all_classes.update(torch.unique(pred).tolist())  # Add classes from predictions
            
            # Sort the classes to ensure order
            all_classes = sorted(all_classes)
            
            # Define class labels dynamically (for simplicity, use numeric labels for now)
            class_labels = {cls: f"Class {cls}" for cls in all_classes}
            num_classes = len(class_labels)
            
            # Create a discrete colormap with exactly `num_classes` colors
            colormap = plt.cm.get_cmap("viridis", num_classes)
            
            # Create legend patches using discrete colors from the colormap
            legend_patches = [
                mpatches.Patch(color=colormap(i), label=f"{i}: {label}")
                for i, label in class_labels.items()
            ]
            
            # Create subplots
            fig, axes = plt.subplots(num_rows, pairs_per_row * 2, figsize=(15, 5 * num_rows))
            
            # If there's only one row, ensure axes is 2D
            if num_rows == 1:
                axes = [axes]
            
            # Flatten axes for easier indexing
            axes = [ax for row_axes in axes for ax in (row_axes if isinstance(row_axes, (list, np.ndarray)) else [row_axes])]
            
            # Plot ground truth and predictions
            for idx, (sample_id, ground_truth, prediction) in enumerate(plot_samples):
                # Calculate column index (each pair takes two columns)
                col_idx = idx * 2
            
                # Ground truth
                axes[col_idx].imshow(ground_truth.cpu().numpy(), cmap="viridis", interpolation="none",
                                     vmin=min(all_classes), vmax=max(all_classes))
                axes[col_idx].set_title(f"Sample {sample_id}\n Ground Truth \n [{ground_truth.shape[0]} x {ground_truth.shape[1]}]")
                axes[col_idx].axis("off")

                # Saving ground truth image
                ground_truth_data = ground_truth.cpu().numpy()  # Convert to NumPy array
                ground_truth_colored = apply_colormap_and_normalize(
                    ground_truth_data, cmap="viridis", vmin=min(all_classes), vmax=max(all_classes)
                )
                ground_truth_img = Image.fromarray(ground_truth_colored)  # Convert to Pillow Image
                ground_truth_img.save(os.path.join(best_trial_dir, f"{sample_id}_gt.png"))
            
                # Prediction
                axes[col_idx + 1].imshow(prediction.cpu().numpy(), cmap="viridis", interpolation="none",
                                         vmin=min(all_classes), vmax=max(all_classes))
                axes[col_idx + 1].set_title(f"Sample {sample_id}\n Prediction \n [{prediction.shape[0]} x {prediction.shape[1]}]")
                axes[col_idx + 1].axis("off")

                # Saving prediction image
                prediction_data = prediction.cpu().numpy()  # Convert to NumPy array
                prediction_colored = apply_colormap_and_normalize(
                    prediction_data, cmap="viridis", vmin=min(all_classes), vmax=max(all_classes)
                )
                prediction_img = Image.fromarray(prediction_colored)  # Convert to Pillow Image
                prediction_img.save(os.path.join(best_trial_dir, f"{sample_id}_pred.png"))
            
            # Hide any unused subplots
            for unused_ax in axes[len(plot_samples) * 2:]:
                unused_ax.axis("off")
            
            # Add legend
            fig.legend(
                handles=legend_patches,
                loc="upper center",  # Places the legend below the plot
                ncol=len(all_classes),
                bbox_to_anchor=(0.5, 0.9),  # Adjusts position below the plot
                fontsize=12
            )
            
            # Add title
            fig.suptitle(
                f"Trial {trial.number}\nGround Truth vs Prediction\n"
                f"Lowest Val Loss: {lowest_val_loss:.4f} - Lowest Epoch: {lowest_epoch}",
                fontsize=16, y=1.10
            )
            
            # Save as SVG
            plt.tight_layout(rect=[0, 0, 1, 0.85])
            result_plot_path = os.path.join(best_trial_dir, f"validation_gt_vs_pred.svg")
            fig.savefig(result_plot_path, format="svg", bbox_inches="tight")
            print(f"Validation results plot saved at: {result_plot_path}")
            plt.close(fig)



        # Log CUDA memory at the start of the trial
        if torch.cuda.is_available():
            trial.set_user_attr("final_allocated_cuda_memory", torch.cuda.memory_allocated() / 1e6)
            trial.set_user_attr("final_reserved_cuda_memory", torch.cuda.memory_reserved() / 1e6)   
    
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        
        return lowest_val_loss

    except torch.cuda.OutOfMemoryError:
        # Log CUDA memory at the start of the trial
        if torch.cuda.is_available():
            trial.set_user_attr("final_allocated_cuda_memory", torch.cuda.memory_allocated() / 1e6)
            trial.set_user_attr("final_reserved_cuda_memory", torch.cuda.memory_reserved() / 1e6)   # Convert to MB
        try:
            del model
        except UnboundLocalError:
            pass  # Model was never created, so no need to delete it
            
        try:
            del trainer
        except UnboundLocalError:
            pass  # Trainer was never created, so no need to delete it
            
        gc.collect()
        torch.cuda.empty_cache()
        print()
        print(f"{'=' * 50}")
        print(f"{'=' * 14} TRIAL {trial.number} SKIPPED - CUDA OUT OF MEMORY ERROR {'=' * 14}")
        print(f"{'=' * 50}")
        print()
        trial.report(float('inf'), step=0)  # Assign a penalty
        # time.sleep(60) # Allow time for the CUDA memory to be released
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":    
    # Run exactly one trial
    study.optimize(objective, n_trials=1)