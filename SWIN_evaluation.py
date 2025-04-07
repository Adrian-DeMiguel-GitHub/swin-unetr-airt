#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# # CHEKING DISK STORAGE

# In[2]:


import shutil

# Get disk space details
total, used, free = shutil.disk_usage("/")

# Convert to human-readable format
def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

print()
print(f"Total Space: {format_size(total)}")
print(f"Used Space: {format_size(used)}")
print(f"Free Space: {format_size(free)}")
print()


# In[3]:


import os

def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            # Add file size, skipping broken symbolic links
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def print_directory_size(directory):
    size_bytes = get_directory_size(directory)
    # Convert bytes to a human-readable format (KB, MB, GB)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            print(f"Size of '{directory}': {size_bytes:.2f} {unit}")
            break
        size_bytes /= 1024

print()
# Example usage
preprocessed_files_path = "data/train_data/preprocessed_files"  # Change this to your target directory
print_directory_size(preprocessed_files_path)

preprocessed_files_path = "executions"  # Change this to your target directory
print_directory_size(preprocessed_files_path)
print()


# # SPECIFY EXECUTION ID TO EVALUATE
# 

# In[4]:


import os

############### EXECUTION SETTINGS ###############

PARENT_EXECUTION_DIR = "executions/swin_unetr/tuning-train-test"

############## EXECUTION ID ######################

EXECUTION_ID = 174 # Update ID

execution_dir = os.path.join(PARENT_EXECUTION_DIR, f"id={EXECUTION_ID}")
print()
print(f"Execution to evaluate: {execution_dir}")
print()


# # LOAD EXECUTION SETTINGS

# In[5]:


import json
import pprint

# Load JSON
info_execution_json_file_path = os.path.join(execution_dir, "info_execution.json")

with open(info_execution_json_file_path, "r") as json_file:
    info_execution = json.load(json_file)

# pprint.pprint(info_execution, sort_dicts=False, width=100)
print(info_execution)
print()

################### DATA SETTINGS #################

DATA_DIR = info_execution["DATA_SETTINGS"]["DATA_DIR"] 
# Directory where the samples (data and labels folders) and the metadata.json file are located.

NUM_WORKERS = info_execution["DATA_SETTINGS"]["NUM_WORKERS"]
# Number of logical CPU Cores used for parallelizing data laoding


# # LOAD DATA SPLITTING INFO

# In[6]:


import json
import os


# Load metadata JSON file
metadata_json_path = os.path.join(info_execution["DATA_SETTINGS"]["DATA_DIR"], info_execution["DATA_SETTINGS"]["METADATA_DATASET"])
with open(metadata_json_path, "r") as f:
    metadata = json.load(f)

# Restore validation files
validation_files_original_copy = {sample: metadata[sample] for sample in info_execution["DATA_SETTINGS"]["VALIDATION_SET"]}

# Restore train-test splits
train_test_splits_orginal_copy = []
for split_key, split_data in info_execution["DATA_SETTINGS"]["TRAIN_TEST_SPLITS"].items():
    train_files = {sample: metadata[sample] for sample in split_data["TRAIN_SET"]}
    test_files = {sample: metadata[sample] for sample in split_data["TEST_SET"]}
    train_test_splits_orginal_copy.append((train_files, test_files))

print("✅ Successfully restored `train_test_splits_orginal_copy` and `validation_files_original_copy` from JSON!")

print()


# # Import SwinUNETR_AIRT LIGHTNING MODEL

# In[7]:


import importlib
from models.swin_unetr import Swin_UNETR_AIRT_lightning_model

# Reload the module (if necessary)
importlib.reload(Swin_UNETR_AIRT_lightning_model)

# Import the class from the reloaded module
from models.swin_unetr.Swin_UNETR_AIRT_lightning_model import SwinUNETR_AIRT_LightningModel
print()


# # Import SwinUNETR_AIRT DATASET

# In[8]:


import importlib
from models.swin_unetr import Swin_UNETR_AIRT_dataset

# Reload the module (if necessary)
importlib.reload(Swin_UNETR_AIRT_dataset)

# Import the class from the reloaded module
from models.swin_unetr.Swin_UNETR_AIRT_dataset import SwinUNETR_AIRT_Dataset
print()


# # Importing libraries

# In[9]:


import os
import json
import time
import shutil
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns  # Used for plotting
from PIL import Image  # Used to save images
from datetime import timedelta, datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import optuna


# # Functions needed

# In[10]:


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


# # BEST TRIAL IN STUDY

# ## Summary

# In[11]:


tuning_dir = os.path.join(execution_dir, 'tuning')
best_trial_json_path = os.path.join(tuning_dir, "best_trial", "best_trial_summary_info.json")

# Load JSON
with open(best_trial_json_path, "r") as json_file:
    best_trial_config = json.load(json_file)

# Extract the best trial number and validation loss
best_trial_info = best_trial_config.get("MODEL_INFO", {})
best_trial_training_config = best_trial_config.get("TRAINING_CONFIG", {})

val_loss = best_trial_config.get("VAL_LOSS")
val_loss = float(val_loss) if isinstance(val_loss, (int, float)) else None  # Convert only if it's a number

print()
print(f"{'=' * 50}")
print(f"{'=' * 18} BEST TRIAL: {best_trial_config.get('TRIAL_ID', 'N/A')} {'=' * 17}")
print(f"{'=' * 16} Val. Loss: {val_loss:.4f} {'=' * 16}" if val_loss is not None else f"{'=' * 16} Val. Loss: N/A {'=' * 16}")
print(f"{'=' * 50}")
print()

print(f"Total Parameters: {best_trial_info.get('model_total_params', 'N/A')}")
print(f"Trainable Parameters: {best_trial_info.get('model_trainable_params', 'N/A')}")
print(f"Non-Trainable Parameters: {best_trial_info.get('model_non_trainable_params', 'N/A')}")
print(f"FLOPs: {best_trial_info.get('model_gflops_inference', 0.0):.3f} GFLOPs")
print()
print()

# Pretty-print the configurations
pprint.pprint(best_trial_config, sort_dicts=False, width=100)

print()


# ## Validation Results (Ground Truth vs Prediction)

# In[12]:


from IPython.display import SVG

svg_file = os.path.join(tuning_dir, "best_trial", f"validation_gt_vs_pred.svg")
# Display the SVG file
SVG(svg_file)  # Replace with your SVG file path


# ## Training vs Validation Loss Evolution

# In[13]:


from IPython.display import SVG

svg_file = os.path.join(tuning_dir, "best_trial", f"train_val_loss_evolution.svg")
# Display the SVG file
SVG(svg_file)  # Replace with your SVG file path


# # EVALUATION

# In[14]:


def update_evaluation_summary_json_file(data):
    """
    Overwrites the JSON file on disk with the updated data.

    Parameters:
    - data (dict): The updated dictionary to write to the file.
    """
    try:
        # Ensure the directory exists
        json_path = os.path.join(execution_dir, "evaluation_summary.json")

        # Write the updated data to the file, overwriting it
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"JSON file successfully updated at: {json_path}")

    except Exception as e:
        print(f"An error occurred while updating the JSON file: {e}")

# Generate the evaluation summary
evaluation_summary = {
    "total_execution_time_seconds": None,
    "splits": [],
    "best_split": {
        "testing_loss": 1
    },
    "average_testing_loss": None,
    "testing_loss_min_max_difference": None,
    "average_testing_mean_iou": None,
    "average_testing_dice_coefficient": None,
    "average_testing_fpr": None,
    "average_validation_loss": None,
    "average_training_loss": None,
    "average_training_time_seconds": None,
    "average_testing_time_seconds": None
}
update_evaluation_summary_json_file(evaluation_summary)

# Measure start training time
start_execution_time = time.time()

print(f"Evaluations starts at: {datetime.now()}\n")
print()

for split_idx, (train_files, test_files) in enumerate(train_test_splits_orginal_copy):

    # Clear the GPU cache between splits
    torch.cuda.empty_cache()

    split_details = {
        "split_index": split_idx,
        "train_samples": None,
        "val_samples": None,
        "test_samples": None,
        "num_train_patches": None,
        "best_epoch": None,
        "training_loss_best_epoch": None,
        "validation_loss_best_epoch": None,
        "val_mean_iou_best_epoch": None,
        "val_dice_best_epoch": None,
        "val_fpr_best_epoch": None,
        "training_time_seconds": None,
        "testing_loss": None,
        "testing_mean_iou": None,
        "testing_dice_coefficient": None,
        "testing_fpr": None,
        "testing_time_seconds": None
    }

    evaluation_summary["splits"].append(split_details)
    update_evaluation_summary_json_file(evaluation_summary)
    
    ####################################################################################
    ############################# DATA PREPARATION #####################################
    ####################################################################################
    
    print()
    print(f"{'=' * 50}")
    print(f"{'=' * 20} TRAIN/TEST SPLIT: {split_idx} {'=' * 20}")
    print(f"{'=' * 16} Time: {datetime.now()} {'=' * 16}")
    print(f"{'=' * 50}")
    print()

    print()
    print(f"{'=' * 15}> TRAINING DATA PREPARATION")
    print()

    print(f" SAMPLES: {train_files.keys()}")
    print()
    print()
    print()
    
    train_dataset = SwinUNETR_AIRT_Dataset(
        is_inference_mode = False,
        augmentation = True,
        metadata_dict_with_files_selected=train_files,
        data_dir=DATA_DIR,
        model_input_dims=best_trial_config["ARCHITECTURE_CONFIG"]["model_input_dimensions"],
        overlap = best_trial_config["TRAINING_CONFIG"]["overlap_training"],
        preprocessing_strategy = best_trial_config["TRAINING_CONFIG"]["pre_processing_strategy"]
        
    )

    print()
    print()
    print(f"{'=' * 15}> VALIDATION DATA PREPARATION")
    print()

    print(f" SAMPLES: {validation_files_original_copy.keys()}")
    print()
    print()
    print()

    validation_files = validation_files_original_copy.copy()
    
    val_dataset = SwinUNETR_AIRT_Dataset(
        is_inference_mode = True,
        augmentation = True,
        metadata_dict_with_files_selected=validation_files,
        data_dir=DATA_DIR,
        model_input_dims=best_trial_config["ARCHITECTURE_CONFIG"]["model_input_dimensions"],
        overlap = best_trial_config["TRAINING_CONFIG"]["overlap_inference"],
        preprocessing_strategy = best_trial_config["TRAINING_CONFIG"]["pre_processing_strategy"]
    )
       
    train_loader = DataLoader(train_dataset, batch_size=best_trial_config["TRAINING_CONFIG"]["batch_size_training"], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=best_trial_config["TRAINING_CONFIG"]["batch_size_inference"], collate_fn=custom_collate, num_workers=NUM_WORKERS)

    print()

    evaluation_summary["splits"][split_idx]["train_samples"] = list(train_files.keys())
    evaluation_summary["splits"][split_idx]["val_samples"] = list(validation_files.keys())
    evaluation_summary["splits"][split_idx]["test_samples"] = list(test_files.keys())
    evaluation_summary["splits"][split_idx]["num_train_patches"] = len(train_dataset)
    update_evaluation_summary_json_file(evaluation_summary)

    ####################################################################################
    ############################# TRAINER SETTING ######################################
    ####################################################################################

    split_dir = os.path.join(execution_dir, f"train_test_split_{split_idx}")
    
    # Model checkpoint callback    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=split_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode='min'
    )
    
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=best_trial_config["TRAINING_CONFIG"]["patience_epochs_early_stopping"],
        mode='min'
    )
    
    # Define logger
    logger = CSVLogger(
        save_dir=split_dir,
        name="",
        version="")
    
    trainer = Trainer(
        max_epochs=best_trial_config["TRAINING_CONFIG"]["max_training_epochs"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1, # log every n batches
        logger=logger,
        precision="16-mixed",  # Enable FP16 mixed precision
        accumulate_grad_batches=best_trial_config["TRAINING_CONFIG"]["num_batches_grad_accumulation"],
        #max_time="00:00:05:00"  # Limit TIME (DD:HH:MM:SS format)
    )
    

    ####################################################################################
    ############################### TRAINING ###########################################
    ####################################################################################

    print()
    print(f"{'=' * 10} TRAINING STARTS {'=' * 10}")
    print()
            
    # Measure start training time
    start_time = time.time()
    
    # Define and initilize model
    model = SwinUNETR_AIRT_LightningModel(
        # Training Configuration
        training_config=best_trial_config["TRAINING_CONFIG"],            
        # Architecture Configuration
        architecture_config=best_trial_config["ARCHITECTURE_CONFIG"],            
        # Attention Configuration
        attention_config=best_trial_config["ATTENTION_CONFIG"],
        # Normalization Configuration
        normalization_config=best_trial_config["NORMALIZATION_CONFIG"],         
        # Regularization Configuration
        regularization_config=best_trial_config["REGULARIZATION_CONFIG"]
    )
    # Fit the model
    trainer.fit(model, train_loader, val_loader)
    
    # Measure end training time
    end_time = time.time()
    
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Training completed in {elapsed_time // 3600:.0f}h {elapsed_time % 3600 // 60:.0f}m {elapsed_time % 60:.0f}s")

    print()
    print(f"{'=' * 10} TRAINING FINISHED {'=' * 10}")
    print()
    
    ####################################################################################
    ################## PLOTTING TRAINING & VALIDATION LOSSES ###########################
    ####################################################################################

    print()
    print(f"{'=' * 10} PLOTTING TRAINING & VALIDATION LOSSES EVOLUTION {'=' * 10}")
    print()
    
    # Path to the latest metrics file
    metrics_file = os.path.join(split_dir, 'metrics.csv')  
    # Load the logged metrics
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
        lowest_epoch = int(lowest_val_loss_row['epoch'])
    
        print(f"Lowest val_loss: {lowest_val_loss} at epoch {lowest_epoch}")
    else:
        print(f"val_loss column not found in {metrics_file}")

    # Handle train_loss and val_loss separately
    train_loss = metrics_df[['epoch', 'train_loss_epoch']].dropna().rename(columns={'train_loss_epoch': 'loss'})
    train_loss['type'] = 'Training Loss'
    
    val_loss = metrics_df[['epoch', 'val_loss_epoch']].dropna().rename(columns={'val_loss_epoch': 'loss'})
    val_loss['type'] = 'Validation Loss'

    evaluation_summary["splits"][split_idx]["best_epoch"] = lowest_epoch
    evaluation_summary["splits"][split_idx]["training_loss_best_epoch"] = metrics_df['train_loss_epoch'].dropna().reset_index(drop=True)[lowest_epoch]
    evaluation_summary["splits"][split_idx]["validation_loss_best_epoch"] = metrics_df['val_loss_epoch'].dropna().reset_index(drop=True)[lowest_epoch]
    evaluation_summary["splits"][split_idx]["val_mean_iou_best_epoch"] = metrics_df['val_mean_iou_epoch'].dropna().reset_index(drop=True)[lowest_epoch]
    evaluation_summary["splits"][split_idx]["val_dice_best_epoch"] = metrics_df['val_dice_epoch'].dropna().reset_index(drop=True)[lowest_epoch]
    evaluation_summary["splits"][split_idx]["val_fpr_best_epoch"] = metrics_df['val_fpr_epoch'].dropna().reset_index(drop=True)[lowest_epoch]
    evaluation_summary["splits"][split_idx]["training_time_seconds"] = elapsed_time
    update_evaluation_summary_json_file(evaluation_summary)
    
    # Combine the two datasets for seaborn
    plot_data = pd.concat([train_loss, val_loss], axis=0)
    
    # Plot using seaborn
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(data=plot_data, x='epoch', y='loss', hue='type', marker="o", style="type", dashes=False)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plot_title_template = (
        f"Lowest Val Loss : {evaluation_summary['splits'][split_idx]['validation_loss_best_epoch']:.4f} - Epoch: {evaluation_summary['splits'][split_idx]['best_epoch']} - "
        f"Number Training Patches: {evaluation_summary['splits'][split_idx]['num_train_patches']}"
    )
    plt.title(f"Split {split_idx} \n Training and Validation Loss Over Epochs \n" + plot_title_template)
    plt.grid(True)
    plt.legend(title="Loss Type")

    # Save as SVG
    plt.tight_layout()
    result_plot_path = os.path.join(split_dir, f"train_val_loss_evolution_split_{split_idx}.svg")
    fig.savefig(result_plot_path, format="svg", bbox_inches="tight")
    print(f"Training and validation loss evolution plot for split {split_idx} saved at: {result_plot_path}")
    plt.show()
    plt.close(fig)
    
    ####################################################################################
    ###################### PLOTTING BEST_CHECKPOINT RESULTS ############################
    ####################################################################################

    validation_folder = os.path.join(split_dir, "validation_results")
    os.makedirs(validation_folder, exist_ok=True)
    
    print()
    print(f"{'=' * 10} PLOTTING BEST_CHECKPOINT (EPOCH) VALIDATION RESULTS {'=' * 10}")
    print(f"{'=' * 10} GROUND TRUTH VS PREDICTIONS {'=' * 10}")
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

        # Saving ground truth image
        ground_truth_data = ground_truth.cpu().numpy()  # Convert to NumPy array
        ground_truth_colored = apply_colormap_and_normalize(
            ground_truth_data, cmap="viridis", vmin=min(all_classes), vmax=max(all_classes)
        )
        ground_truth_img = Image.fromarray(ground_truth_colored)  # Convert to Pillow Image
        ground_truth_img.save(os.path.join(validation_folder, f"{sample_id}_gt.png"))
    
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
        prediction_img.save(os.path.join(validation_folder, f"{sample_id}_pred.png"))
    
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

    # Titles for the plots
    plot_title_template = (
        f"Lowest Val Loss : {evaluation_summary['splits'][split_idx]['validation_loss_best_epoch']:.4f} - Epoch: {evaluation_summary['splits'][split_idx]['best_epoch']}\n"
        f"Number Training Patches: {evaluation_summary['splits'][split_idx]['num_train_patches']}\n"
        f"Pixel-Wise Metrics: Val IoU: { evaluation_summary['splits'][split_idx]['val_mean_iou_best_epoch']:.4f} - "
        f"Val F1-Score: {evaluation_summary['splits'][split_idx]['val_dice_best_epoch']:.4f} - "
        f"Val FPR: {evaluation_summary['splits'][split_idx]['val_fpr_best_epoch']:.4f}"
    )
    
    # Add title
    fig.suptitle(
        f"Split {split_idx} (Validation Results)\nGround Truth vs Prediction\n" + plot_title_template,
        fontsize=16, y=1.10
    )
    
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    result_plot_path = os.path.join(validation_folder, f"validation_gt_vs_pred_split_{split_idx}.svg")
    fig.savefig(result_plot_path, format="svg", bbox_inches="tight")
    print(f"Validation results plot for split {split_idx} saved at: {result_plot_path}")
    plt.show()
    plt.close(fig)
    

    ############################################################################################
    ################################### TESTING DATA PREPARATION ###############################
    ############################################################################################
    
    print()
    print(f"{'=' * 10} TESTING DATA PREPARATION {'=' * 10}")
    print()
    
    print(f" SAMPLES: {test_files.keys()}")
    print()
    
    test_dataset = SwinUNETR_AIRT_Dataset(
        is_inference_mode = True,
        augmentation = False,
        metadata_dict_with_files_selected=test_files,
        data_dir=DATA_DIR,
        model_input_dims=best_trial_config["ARCHITECTURE_CONFIG"]["model_input_dimensions"],
        overlap = best_trial_config["TRAINING_CONFIG"]["overlap_inference"],
        preprocessing_strategy = best_trial_config["TRAINING_CONFIG"]["pre_processing_strategy"]
    )
       
    test_loader = DataLoader(test_dataset, batch_size=best_trial_config["TRAINING_CONFIG"]["batch_size_inference"], collate_fn=custom_collate, num_workers=NUM_WORKERS)

    ####################################################################################
    ############################## TESTING #############################################
    ####################################################################################
    
    print()
    print(f"{'=' * 10} TESTING STARTS {'=' * 10}")
    print()
            
    # Measure start training time
    start_time = time.time()
    
    # Find all .ckpt files in the directory
    ckpt_files = [filename for filename in os.listdir(split_dir) if filename.endswith('.ckpt')]
    
    # Handle different cases
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in the directory: {split_dir}")
    elif len(ckpt_files) > 1:
        raise RuntimeError(f"Multiple .ckpt files found in the directory: {split_dir} -> {ckpt_files}")
    else:
        model_checkpoint_path = os.path.join(split_dir, ckpt_files[0])
        print(f"Found checkpoint: {model_checkpoint_path}")
    
    # Load the best model
    best_model = SwinUNETR_AIRT_LightningModel.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path,
        
        # Training Configuration
        training_config=best_trial_config["TRAINING_CONFIG"],            
        # Architecture Configuration
        architecture_config=best_trial_config["ARCHITECTURE_CONFIG"],            
        # Attention Configuration
        attention_config=best_trial_config["ATTENTION_CONFIG"],
        # Normalization Configuration
        normalization_config=best_trial_config["NORMALIZATION_CONFIG"],         
        # Regularization Configuration
        regularization_config=best_trial_config["REGULARIZATION_CONFIG"]
    )
    
    test_results = trainer.test(best_model, dataloaders=test_loader)[0]
    
    # Measure end training time
    end_time = time.time()
    
    # Calculate and print the elapsed time
    testing_elapsed_time = end_time - start_time
    
    print(f"Testing completed in {testing_elapsed_time // 3600:.0f}h {testing_elapsed_time % 3600 // 60:.0f}m {testing_elapsed_time % 60:.0f}s")
    
    print()
    print(f"{'=' * 10} TESTING FINISHED {'=' * 10}")
    print()

    evaluation_summary["splits"][split_idx]["testing_loss"] = test_results["test_loss_epoch"]
    evaluation_summary["splits"][split_idx]["testing_mean_iou"] = test_results["test_mean_iou_epoch"]
    evaluation_summary["splits"][split_idx]["testing_dice_coefficient"] = test_results["test_dice_epoch"]
    evaluation_summary["splits"][split_idx]["testing_fpr"] = test_results["test_fpr_epoch"]
    evaluation_summary["splits"][split_idx]["testing_time_seconds"] = testing_elapsed_time
    update_evaluation_summary_json_file(evaluation_summary)

    ####################################################################################
    ############################## PLOTTING TEST RESULTS ###############################
    ####################################################################################

    testing_folder = os.path.join(split_dir, "testing_results")
    os.makedirs(testing_folder, exist_ok=True)
    
    print()
    print(f"{'=' * 10} PLOTTING TEST RESULTS {'=' * 10}")
    print(f"{'=' * 10} GROUND TRUTH VS PREDICTIONS {'=' * 10}")
    print()

    # Control the maximum number of samples to show
    num_samples_to_show = 9
    
    # Adjust the number of samples if fewer samples are available
    available_samples = list(zip(
        best_model.test_sample_ids,
        best_model.test_ground_truths,
        best_model.test_predictions
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
        ground_truth_img.save(os.path.join(testing_folder, f"{sample_id}_gt.png"))
    
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
        prediction_img.save(os.path.join(testing_folder, f"{sample_id}_pred.png"))
    
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

    # Titles for the plots
    plot_title_template = (
        f"Test Loss : {evaluation_summary['splits'][split_idx]['testing_loss']:.4f}\n"
        f"Number Testing Patches: {len(test_dataset)}\n"
        f"Pixel-Wise Metrics: Test IoU: { evaluation_summary['splits'][split_idx]['testing_mean_iou']:.4f} - "
        f"Test F1-Score: {evaluation_summary['splits'][split_idx]['testing_dice_coefficient']:.4f} - "
        f"Test FPR: {evaluation_summary['splits'][split_idx]['testing_fpr']:.4f}"
    )
    
    # Add title
    fig.suptitle(
        f"Split {split_idx} (Testing Results)\nGround Truth vs Prediction\n" + plot_title_template,
        fontsize=16, y=1.10
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    result_plot_path = os.path.join(testing_folder, f"testing_gt_vs_pred_split_{split_idx}.svg")
    fig.savefig(result_plot_path, format="svg", bbox_inches="tight")
    print(f"Testing results plot for split {split_idx} saved at: {result_plot_path}")
    plt.show()
    plt.close(fig)
    
    del model
    del best_model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()    

    ####################################################################################
    ############################ SPLIT PERFORMANCE COMPARISON ###########################
    ####################################################################################
    
    if evaluation_summary["splits"][split_idx]["testing_loss"] < evaluation_summary["best_split"]["testing_loss"]:
        evaluation_summary["best_split"] = evaluation_summary["splits"][split_idx]

# Measure end training time
end_execution_time = time.time()
    
# Calculate and print the elapsed time
elapsed_execution_time = end_execution_time - start_execution_time

####################################################################################
############################ ANALYSING SPLIT PERFORMANCE ###########################
####################################################################################

testing_losses_splits = [split["testing_loss"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_loss_min_value = np.min(testing_losses_splits)
testing_loss_max_value = np.max(testing_losses_splits)
testing_loss_min_max_difference = testing_loss_max_value - testing_loss_min_value
testing_mean_iou_splits = [split["testing_mean_iou"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_dice_coeffcient_splits = [split["testing_dice_coefficient"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_fpr_splits = [split["testing_fpr"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
training_losses_splits = [split["training_loss_best_epoch"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
validation_losses_splits = [split["validation_loss_best_epoch"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
training_times_splits = [split["training_time_seconds"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_times_splits = [split["testing_time_seconds"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]

# Update the evaluation summary
evaluation_summary["total_execution_time_seconds"] = elapsed_execution_time
evaluation_summary["average_testing_loss"] = np.mean(testing_losses_splits)
evaluation_summary["testing_loss_min_max_difference"] = testing_loss_min_max_difference
evaluation_summary["average_testing_mean_iou"] = np.mean(testing_mean_iou_splits)
evaluation_summary["average_testing_dice_coefficient"] = np.mean(testing_dice_coeffcient_splits)
evaluation_summary["average_testing_fpr"] = np.mean(testing_fpr_splits)
evaluation_summary["average_validation_loss"] = np.mean(validation_losses_splits)
evaluation_summary["average_training_loss"] = np.mean(training_losses_splits)
evaluation_summary["average_training_time_seconds"] = np.mean(training_times_splits)
evaluation_summary["average_testing_time_seconds"] = np.mean(testing_times_splits)
update_evaluation_summary_json_file(evaluation_summary)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Extract data for plotting
split_indexes = [split["split_index"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_samples_splits = [len(split["test_samples"]) for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
training_samples_splits = [len(split["train_samples"]) for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
validation_samples_splits = [len(split["val_samples"]) for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]

# Prepare the data for Seaborn
data = pd.DataFrame({
    "Split Index": split_indexes,
    "Training Instances": training_samples_splits,
    "Validation Instances": validation_samples_splits,
    "Test Instances": testing_samples_splits,
})

# Melt the data for easier plotting with Seaborn
data_melted = data.melt(id_vars="Split Index", 
                        var_name="Set Type", 
                        value_name="Instances")

# Plot the bar chart using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(
    data=data_melted,
    x="Split Index",
    y="Instances",
    hue="Set Type",
    dodge=True,  # Ensure separate bars for each loss type
)

# Add labels and title
plt.xlabel("Split Index")
plt.ylabel("Instances")
plt.title("Sets Comparison")
plt.legend(title="Set", loc="upper right")
plt.grid(linestyle="--", alpha=0.7)

# Save the plot as an SVG file
plot_path_svg = os.path.join(execution_dir, "sets_comparison.svg")
plt.savefig(plot_path_svg, format="svg", bbox_inches="tight")
print(f"Bar plot of Evaluation Loss Comparison saved at: {plot_path_svg}")

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Extract data for plotting
split_indexes = [split["split_index"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_losses_splits = [split["testing_loss"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
training_losses_splits = [split["training_loss_best_epoch"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
validation_losses_splits = [split["validation_loss_best_epoch"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]

# Prepare the data for Seaborn
data = pd.DataFrame({
    "Split Index": split_indexes,
    "Training Loss": training_losses_splits,
    "Validation Loss": validation_losses_splits,
    "Test Loss": testing_losses_splits,
})

# Melt the data for easier plotting with Seaborn
data_melted = data.melt(id_vars="Split Index", 
                        var_name="Loss Type", 
                        value_name="Loss")

# Plot the bar chart using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(
    data=data_melted,
    x="Split Index",
    y="Loss",
    hue="Loss Type",
    dodge=True,  # Ensure separate bars for each loss type
)

# Add labels and title
plt.xlabel("Split Index")
plt.ylabel("Loss")
plt.title("Evaluation Loss Comparison")
plt.legend(title="Loss Type", loc="upper left")
plt.grid(linestyle="--", alpha=0.7)

# Save the plot as an SVG file
plot_path_svg = os.path.join(execution_dir, "evaluation_loss_comparison.svg")
plt.savefig(plot_path_svg, format="svg", bbox_inches="tight")
print(f"Bar plot of Evaluation Loss Comparison saved at: {plot_path_svg}")

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

training_times_splits = [split["training_time_seconds"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_times_splits = [split["testing_time_seconds"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]

# Prepare the data for Seaborn
data = pd.DataFrame({
    "Split Index": split_indexes,
    "Training Time (s)": training_times_splits,
    "Testing Time (s)": testing_times_splits,
})

# Melt the data for easier plotting with Seaborn
data_melted = data.melt(id_vars="Split Index", 
                        var_name="Time Type", 
                        value_name="Time (s)")

# Plot the bar chart using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(
    data=data_melted,
    x="Split Index",
    y="Time (s)",
    hue="Time Type",
    dodge=True,  # Ensure separate bars for each time type
)

# Add labels and title
plt.xlabel("Split Index")
plt.ylabel("Time (s)")
plt.title("Training and Testing Time Comparison")
plt.legend(title="Time Type", loc="upper right")
plt.grid(linestyle="--", alpha=0.7)

# Save the plot as an SVG file
plot_path_svg = os.path.join(execution_dir, "evaluation_training_testing_time_comparison.svg")
plt.savefig(plot_path_svg, format="svg", bbox_inches="tight")
print(f"Bar plot of Training and Testing Time Comparison saved at: {plot_path_svg}")

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:


##################################### PLOTTING #####################################

best_split = evaluation_summary["best_split"]["split_index"]
best_split_testing_loss = evaluation_summary["best_split"]["testing_loss"]
validation_losses_splits = [split["validation_loss_best_epoch"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_losses_splits = [split["testing_loss"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_mean_iou_splits = [split["testing_mean_iou"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_dice_splits = [split["testing_dice_coefficient"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]
testing_fpr_splits = [split["testing_fpr"] for split in sorted(evaluation_summary["splits"], key=lambda x: x["split_index"])]


# Number of splits
splits = np.arange(len(training_times_splits))

# Create the figure and axis for the combined plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for Training Times
bar_width = 0.35  # Width of the bars
bar_offset = bar_width / 2
ax1.bar(splits - bar_offset, training_times_splits, color='lightgrey', width=bar_width, label='Training Time (s)')
ax1.bar(splits + bar_offset, testing_times_splits, color='lightblue', width=bar_width, label='Testing Time (s)')
ax1.set_xlabel("Split Index")
ax1.set_ylabel("Time (seconds)", color='grey')
ax1.tick_params(axis='y', labelcolor='grey')

# Line plot for Validation Losses
ax2 = ax1.twinx()
ax2.plot(splits, validation_losses_splits, marker='o', color='blue', label='Validation Loss', zorder=3)
ax2.plot(splits, testing_losses_splits, marker='s', color='green', label='Testing Loss', zorder=3)
ax2.scatter(best_split, best_split_testing_loss, color='red', s=120, zorder=4, label=f'Best Split (Split {best_split})')
ax2.set_ylabel("Loss", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Compute mean and standard deviation
avg_validation_loss = np.mean(validation_losses_splits)
avg_testing_loss = np.mean(testing_losses_splits)
avg_testing_mean_iou = np.mean(testing_mean_iou_splits)
avg_testing_dice = np.mean(testing_dice_splits)
avg_testing_fpr = np.mean(testing_fpr_splits)
# Calculate testing loss min, max, and their difference
testing_loss_min_value = np.min(testing_losses_splits)
testing_loss_max_value = np.max(testing_losses_splits)
testing_loss_min_max_difference = testing_loss_max_value - testing_loss_min_value

plot_title = (
    "Final Evaluation (Split Comparison)\n"
    "Training Times and Validation Losses Across Splits\n"
    f"Complete Execution Time (Including Plotting but no Hypertunning): {elapsed_execution_time // 3600:.0f}h {elapsed_execution_time % 3600 // 60:.0f}m {elapsed_execution_time % 60:.0f}s\n"
    f"Avg. Test Loss: {avg_testing_loss:.4f} - Min/Max Difference: {testing_loss_min_max_difference:.4f}\n"
    f"Avg. Test IoU: {avg_testing_mean_iou:.4f} - Avg. Test F1-Score: {avg_testing_dice:.4f} - Avg. Test FPR: {avg_testing_fpr:.4f}\n"
    f"Averaged Val Loss: {avg_validation_loss:.4f}"
)

# Title and Grid
fig.suptitle(plot_title,)
fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.017, 1.00))
ax1.grid(True)

# Save the combined plot
plot_path = os.path.join(execution_dir, f"final_evaluation.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
    
plt.close(fig)

