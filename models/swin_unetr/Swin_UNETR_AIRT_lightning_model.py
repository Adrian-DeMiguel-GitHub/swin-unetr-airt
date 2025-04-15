# PyTorch and PyTorch Lightning
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

# Numpy
import numpy as np

# Optimizer
from torch.optim import AdamW
#from torch.optim.lr_scheduler import CosineAnnealingLR

# MONAI (for medical imaging-related losses and metrics like DiceLoss, MeanIoU, and DiceMetric)
from monai.losses import DiceLoss
from monai.metrics import MeanIoU, DiceMetric

from torchmetrics.classification import MulticlassConfusionMatrix

import importlib

# Import and reload the LinearWarmupCosineAnnealingLR class
from utils import lr_schedulers
importlib.reload(lr_schedulers)
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR

# Import and reload the SwinUNETR_AIRT class
from models.swin_unetr import Swin_UNETR_AIRT_architecture
importlib.reload(Swin_UNETR_AIRT_architecture)
from models.swin_unetr.Swin_UNETR_AIRT_architecture import SwinUNETR_AIRT


# Define a PyTorch Lightning model wrapper
class SwinUNETR_AIRT_LightningModel(pl.LightningModule):
    def __init__(self, training_config: dict, architecture_config: dict, attention_config: dict, normalization_config: dict, regularization_config: dict):
        super(SwinUNETR_AIRT_LightningModel, self).__init__()

        ################# CONFIGURATIONS #################

        self.training_config = training_config
        self.architecture_config = architecture_config
        
        ###################### MODEL INSTANTIATION ###########################

        self.model = SwinUNETR_AIRT(architecture_config=self.architecture_config, attention_config=attention_config, normalization_config=normalization_config, regularization_config=regularization_config).to("cuda")      
        
        ###################### LOGGING PARAMETERS ###########################

        self.enable_batch_logging_into_console = False  # Boolean flag for controlling batch logging into console
        self.enable_training_epoch_logging_into_console = True  # Boolean flag for controlling training epoch logging into console
        self.enable_validation_epoch_logging_into_console = True  # Boolean flag for controlling validation epoch logging into console
        self.enable_testing_epoch_logging_into_console = True  # Boolean flag for controlling testing epoch logging into console

        ###################### LOSS & OTHER METRICS ###########################

        self.include_background_in_loss_and_metrics = False  # Boolean flag for controlling inclusion of background in loss and metrics

        # DiceLoss ask that the model's output must have at least two channels.
        # The first channel is assumed to represent the background class,
        # while subsequent channels represent different foreground classes.
        #
        # If include_background=False, the Dice loss computation will exclude the first channel
        # (i.e., it will not compute the Dice score for the first class).
        # The loss will be computed only for the rest of the channels.
        #
        # DiceLoss does not apply any activation function by default (this is our case)
        # Therefore, we have to apply the activation function (in our case, softmax) before computing the loss
        # over the model's output (y_hat)
        #
        # y_hat (prediction) is expected to be a multiple-channel tensor containing,
        # in each channel/class, the probability corresponding to the channel/class for each pixel/voxel (i.e. the probability that pixel belongs to the class represented by that channel)
        # (I.e. y_hat shape: (OUTPUT_CHANNELS, HEIGHT, WIDTH) )
        # The first channel is assumed to represent the background class, if include_background=True.
        #
        # y (ground truth) is expected to be a multiple-channel tensor where each channel represents
        # a class, and the grid corresponding to that channel has to be one-hot encoded (just 1s and 0s) representing when
        # the corrresponding pixel is labeled with that class (1) or not (0).
        # (I.e. y shape: (OUTPUT_CHANNELS, HEIGHT, WIDTH) )
        # The first channel is assumed to represent the background class, if include_background=True.

        self.loss_fn = DiceLoss(include_background=self.include_background_in_loss_and_metrics)

        # When using include_background=True in MeanIoU,
        # the model's output must have at least two channels.
        # The first channel is assumed to represent the background class,
        # while subsequent channels represent different foreground classes.
        #
        # Since reduction="mean", MeanIoU will return the average IoU score across all classes,
        # including the background.
        #
        # y_hat (prediction) is expected to be a multiple-channel tensor. It must be one-hot format and first dim is batch. The values should be binarized.
        # (I.e. y_hat shape: (OUTPUT_CHANNELS, HEIGHT, WIDTH) )
        # The first channel is assumed to represent the background class, if include_background=True.
        #
        # y (ground truth) is expected to be a multiple-channel tensor where each channel represents
        # a class, and the grid corresponding to that channel has to be one-hot encoded (just 1s and 0s) representing when
        # the corrresponding pixel is labeled with that class (1) or not (0).
        # (I.e. y shape: (OUTPUT_CHANNELS, HEIGHT, WIDTH) )
        # The first channel is assumed to represent the background class, if include_background=True.

        self.mean_iou_metric = MeanIoU(include_background=self.include_background_in_loss_and_metrics, reduction="none")

        # When using include_background=True in DiceMetric,
        # the model's output must have at least two channels.
        # The first channel is assumed to represent the background class,
        # while subsequent channels represent different foreground classes.
        #
        # Since reduction="mean", DiceMetric will return the average Dice coefficent across all classes,
        # including the background (since include_background=True).
        #
        # y_hat (prediction) is expected to be a multiple-channel tensor. It must be one-hot format and first dim is batch. The values should be binarized.
        # (I.e. y_hat shape: (OUTPUT_CHANNELS, HEIGHT, WIDTH) )
        # The first channel is assumed to represent the background class, if include_background=True.
        #
        # y (ground truth) is expected to be a multiple-channel tensor where each channel represents
        # a class, and the grid corresponding to that channel has to be one-hot encoded (just 1s and 0s) representing when
        # the corrresponding pixel is labeled with that class (1) or not (0).
        # (I.e. y shape: (OUTPUT_CHANNELS, HEIGHT, WIDTH) )
        # The first channel is assumed to represent the background class, if include_background=True.

        self.dice_metric = DiceMetric(include_background=self.include_background_in_loss_and_metrics, reduction="none")

        self.training_batch_losses_in_epoch = []  # Store losses for training batches in training epoch

        self.validation_batch_losses_in_epoch = []  # Store losses for validation batches in validation epoch
        self.validation_batch_mean_ious_in_epoch = []  # Store mean_ious for validation batches in validation epoch
        self.validation_batch_dice_coeffs_in_epoch = []  # Store dice_coeffs for validation batches in validation epoch
        self.validation_batch_fprs_in_epoch = []

        self.testing_batch_losses_in_epoch = []  # Store losses for testing batches in testing epoch
        self.testing_batch_mean_ious_in_epoch = []  # Store mean_ious for testing batches in testing epoch
        self.testing_batch_dice_coeffs_in_epoch = []  # Store dice_coeffs for testing batches in testing epoch
        self.testing_batch_fprs_in_epoch = []
        self.testing_batch_tpr_recalls_in_epoch = []
        self.testing_batch_precisions_in_epoch = []

        self.val_ground_truths = []
        self.val_predictions = []
        self.val_sample_ids = []
        
        self.test_ground_truths = []
        self.test_predictions = []
        self.test_sample_ids = []
        self.test_predictions_probabilities = []

        self.best_epoch_val_loss = float('inf')
        self.best_epoch_val_ground_truths = []
        self.best_epoch_val_predictions = []
        self.best_epoch_val_sample_ids = []

    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        # Optional logging for debugging
        if self.enable_batch_logging_into_console and self.enable_training_epoch_logging_into_console:
            print(f"Training: Epoch {self.current_epoch}, Batch {batch_idx}")
    
        # Unpack the batch
        x, y = batch  # x shape: (batch_size, input_channels, height, width, depth), y shape: (batch_size, output_channels, height, width)
    
        # Forward pass through the model
        y_hat = self(x)  # y_hat shape: (batch_size, output_channels, height, width, 1)
    
        # Apply post-processing (e.g., softmax for probabilities)
        y_hat_probabilities = F.softmax(y_hat, dim=1)
    
        # Compute loss using the provided loss function
        loss = self.loss_fn(y_hat_probabilities, y)
    
        # Log training loss
        self.log(
            'train_loss', 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=False, 
            batch_size=self.trainer.train_dataloader.batch_size
        )
    
        # Keep track of batch losses for custom epoch-level logging
        self.training_batch_losses_in_epoch.append(loss.item())
    
        return loss  # Return loss for Lightning to handle the backward pass

    def validation_step(self, batch, batch_idx):
        
        if self.enable_batch_logging_into_console and self.enable_validation_epoch_logging_into_console:
            print(f"Validation: Epoch {self.current_epoch}, Batch {batch_idx}")

        ids, x, y = batch  # x format: list of sampel_ids (one per sample in the batch), list of dict_patches (one per sample in the batch), y format: list of tuples (y_filename,y_tensor_shape) (one per sample in the batch)

        losses_batch = []  # Use a Python list to collect losses
        mean_ious_batch = [] # Use a Python list to collect losses
        dice_coeffs_batch = [] # Use a Python list to collect losses
        fprs_batch = [] # Use a Python list to collect losses
        

        #############################   COMPUTING PREDICTIONS   #############################################

        for sample_id, sample_x, sample_y in zip(ids, x, y):
            
            ##############  LOADING Y   ###############
            y_filename = sample_y[0]
            y_tensor_shape = sample_y[1]
            # Load the raw data into a NumPy array
            loaded_array_y = np.fromfile(y_filename, dtype=np.float32)           
            # Reshape the loaded array to the original shape
            loaded_array_y = loaded_array_y.reshape(y_tensor_shape)           
            # Convert back to a PyTorch tensor
            y = torch.from_numpy(loaded_array_y).to(self.device)
            ##########################################
            
            #y = torch.load(sample_y, weights_only=True).to(self.device)  # Move to correct device
            y = y.unsqueeze(0)  # Add 1 dimension to match model output format

            # Create a tensor with all values set to NaN
            y_hat = torch.empty_like(y).fill_(float('nan')).to(self.device)
            # Create counter grid [height, width] to check how many patches contribute to each pixel
            y_hat_weighted_avg_method = torch.zeros_like(y).to(self.device)
            counter_grid = np.zeros((y_hat_weighted_avg_method.shape[2], y_hat_weighted_avg_method.shape[3]))
            
            patches_dict_info = sample_x

            for patch_key in list(patches_dict_info.keys()):
                patch_coordinates = patches_dict_info[patch_key]["patch_coord"]
                patch_path = patches_dict_info[patch_key]["patch_path"]

                ##############  LOADING patch_tensor   ###############
                patch_tensor_shape = patches_dict_info[patch_key]["patch_tensor_shape"]
                # Load the raw data into a NumPy array
                loaded_array_patch_tensor = np.fromfile(patch_path, dtype=np.float32)           
                # Reshape the loaded array to the original shape
                loaded_array_patch_tensor = loaded_array_patch_tensor.reshape(patch_tensor_shape)           
                # Convert back to a PyTorch tensor
                patch_tensor = torch.from_numpy(loaded_array_patch_tensor).to(self.device)
                ##########################################
                               
                #patch_tensor = torch.load(patch_path, weights_only=True).to(self.device)
                patch_tensor = patch_tensor.unsqueeze(0)  # Match model input format
                patch_prediction = self(patch_tensor)

                # Fill y_hat with patch predictions
                for i in range(patch_coordinates[0], patch_coordinates[0] + self.architecture_config["model_input_dimensions"][0]):
                    for j in range(patch_coordinates[1], patch_coordinates[1] + self.architecture_config["model_input_dimensions"][1]):
                        if self.training_config["post_processing_strategy"] == "fill_first":
                            is_pixel_without_prediction = torch.isnan(y_hat[:, :, i, j]).any()
                            if is_pixel_without_prediction:
                                y_hat[:, :, i, j] = patch_prediction[:, :, i - patch_coordinates[0], j - patch_coordinates[1]]
                        elif self.training_config["post_processing_strategy"] == "weighted_average":
                            y_hat_weighted_avg_method[:, :, i, j] += patch_prediction[:, :, i - patch_coordinates[0], j - patch_coordinates[1]]
                            counter_grid[i, j] += 1
                            y_hat[:, :, i, j] = y_hat_weighted_avg_method[:, :, i, j] / counter_grid[i, j]
                        else:
                            raise ValueError(
                                f"Invalid configuration value: {self.training_config['post_processing_strategy']}. "
                                f"Allowed values are: ['fill_first', 'weighted_average']"
                            )
                                

            ##############  POST-PROCESSING   ###############
            
            # Apply softmax to probabilities
            y_hat_probabilities = F.softmax(y_hat, dim=1)

            ##############  LOSS   ###############
            
            # Compute loss and store it in the list
            loss = self.loss_fn(y_hat_probabilities, y)
            losses_batch.append(loss.item())

            ############## (METRICS) POST-PROCESSING   ###############

            # Convert predictions to one-hot encoding
            y_hat_one_hot = torch.zeros_like(y_hat_probabilities)
            max_indices = torch.argmax(y_hat_probabilities, dim=1)
            y_hat_one_hot.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)

            ##############  MEAN IOU METRIC   ###############

            # Compute Mean IoU 
            iou = self.mean_iou_metric(y_pred=y_hat_one_hot, y=y)  # Directly call MeanIoU
            mean_iou = iou.mean()  # Mean IoU across classes
            mean_ious_batch.append(mean_iou.item())

            ##############  DICE COEFF METRIC   ###############

            # Compute Dice Metric (Dice Coeff) 
            dice = self.dice_metric(y_pred=y_hat_one_hot, y=y)  # Directly call DiceMetric
            mean_dice = dice.mean()  # Average Dice across classes
            dice_coeffs_batch.append(mean_dice.item())

            ##############  FPR METRIC   ###############

            # Compute FPR Metric
            # Step 1: Classify each pixel accoring to probabilties assigning the class with higher probabilty and flatten it
            y_hat_categorized_flattened = torch.argmax(y_hat_probabilities, dim=1).view(-1)  # Shape: (batch_size * height * width)
            # Step 2: Convert y one_encoded into categorized tensor and flatten it
            y_categorized_flattened = torch.argmax(y, dim=1).view(-1)  # Shape: (batch_size * height * width)
            # Step 3: Define the number of classes
            num_classes = y_hat_probabilities.shape[1]
            # Step 4: Compute the confusion matrix using torchmetrics
            confmat = MulticlassConfusionMatrix(num_classes=y_hat_probabilities.shape[1]).to("cuda")
            confusion_matrix = confmat(y_hat_categorized_flattened, y_categorized_flattened)
            # Step 5: Calculate False Positives (FP) and True Negatives (TN)
            fp = confusion_matrix.sum(dim=0) - torch.diag(confusion_matrix)  # False Positives for each class
            tn = confusion_matrix.sum() - (confusion_matrix.sum(dim=1) + fp)  # True Negatives for each class
            # Step 6: Calculate FPR for each class
            fpr = fp / (fp + tn)
            # Step 7: Exclude the background class (class 0) (if activated)
            if not self.include_background_in_loss_and_metrics:
                fpr = fpr[1:]
            avg_fpr_classes = fpr.mean()
            fprs_batch.append(avg_fpr_classes.item())

            ############## (FINAL OUTPUT SEGMENTATION) POST-PROCESSING   ###############

            # Classify each pixel accoring to probabilties assigning the class with higher probabilty
            y_hat_categorized = torch.argmax(y_hat_probabilities.squeeze(), dim=0)
            # Convert y one_encoded into categorized tensor
            y_categorized = torch.argmax(y.squeeze(), dim=0)
    
            # Save categorized ground_truth and prediction
            self.val_sample_ids.append(sample_id)
            self.val_ground_truths.append(y_categorized)
            self.val_predictions.append(y_hat_categorized)
            

        #############################   LOSS   #############################################
        
        # Calculate average loss for the batch
        batch_loss_avg = np.mean(losses_batch)
        self.log('val_loss', batch_loss_avg, on_step=True, on_epoch=True,prog_bar=False, batch_size=len(x))

        if self.enable_batch_logging_into_console and self.enable_validation_epoch_logging_into_console:
            print(f"- Dice Loss: {batch_loss_avg:.6f}")

        self.validation_batch_losses_in_epoch.append(batch_loss_avg)

        ############################   METRICS   ############################################

        # Calculate average dice_coeff for the batch
        batch_mean_iou_avg = np.mean(mean_ious_batch)
        self.log('val_mean_iou', batch_mean_iou_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.validation_batch_mean_ious_in_epoch.append(batch_mean_iou_avg)        
        

        # Calculate average loss for the batch
        batch_dice_avg = np.mean(dice_coeffs_batch)
        self.log('val_dice', batch_dice_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.validation_batch_dice_coeffs_in_epoch.append(dice_coeffs_batch)

        batch_fpr_avg = np.mean(fprs_batch)
        self.log('val_fpr', batch_fpr_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.validation_batch_fprs_in_epoch.append(batch_fpr_avg) 

        # Return the average batch loss
        return batch_loss_avg



    def test_step(self, batch, batch_idx):
        
        if self.enable_batch_logging_into_console and self.enable_testing_epoch_logging_into_console:
            print(f"Validation: Epoch {self.current_epoch}, Batch {batch_idx}")

        ids, x, y = batch  # x format: list of sampel_ids (one per sample in the batch), list of dict_patches (one per sample in the batch), y format: list of label_paths (one per sample in the batch)

        losses_batch = []  # Use a Python list to collect losses
        mean_ious_batch = [] # Use a Python list to collect losses
        dice_coeffs_batch = [] # Use a Python list to collect losses
        fprs_batch = []
        tpr_recalls_batch = []
        precisions_batch = []
        

        #############################   COMPUTING PREDICTIONS   #############################################

        for sample_id, sample_x, sample_y in zip(ids, x, y):

            ##############  LOADING Y   ###############
            y_filename = sample_y[0]
            y_tensor_shape = sample_y[1]
            # Load the raw data into a NumPy array
            loaded_array_y = np.fromfile(y_filename, dtype=np.float32)           
            # Reshape the loaded array to the original shape
            loaded_array_y = loaded_array_y.reshape(y_tensor_shape)           
            # Convert back to a PyTorch tensor
            y = torch.from_numpy(loaded_array_y).to(self.device)
            ##########################################
            
            #y = torch.load(sample_y, weights_only=True).to(self.device)  # Move to correct device
            y = y.unsqueeze(0)  # Add 1 dimension to match model output format

            # Create a tensor with all values set to NaN
            y_hat = torch.empty_like(y).fill_(float('nan')).to(self.device)
            # Create counter grid [height, width] to check how many patches contribute to each pixel
            y_hat_weighted_avg_method = torch.zeros_like(y).to(self.device)
            counter_grid = np.zeros((y_hat_weighted_avg_method.shape[2], y_hat_weighted_avg_method.shape[3]))
            
            patches_dict_info = sample_x

            for patch_key in list(patches_dict_info.keys()):
                patch_coordinates = patches_dict_info[patch_key]["patch_coord"]
                patch_path = patches_dict_info[patch_key]["patch_path"]

                ##############  LOADING patch_tensor   ###############
                patch_tensor_shape = patches_dict_info[patch_key]["patch_tensor_shape"]
                # Load the raw data into a NumPy array
                loaded_array_patch_tensor = np.fromfile(patch_path, dtype=np.float32)           
                # Reshape the loaded array to the original shape
                loaded_array_patch_tensor = loaded_array_patch_tensor.reshape(patch_tensor_shape)           
                # Convert back to a PyTorch tensor
                patch_tensor = torch.from_numpy(loaded_array_patch_tensor).to(self.device)
                ##########################################
                
                #patch_tensor = torch.load(patch_path, weights_only=True).to(self.device)
                patch_tensor = patch_tensor.unsqueeze(0)  # Match model input format
                patch_prediction = self(patch_tensor)

                # Fill y_hat with patch predictions
                for i in range(patch_coordinates[0], patch_coordinates[0] + self.architecture_config["model_input_dimensions"][0]):
                    for j in range(patch_coordinates[1], patch_coordinates[1] + self.architecture_config["model_input_dimensions"][1]):
                        if self.training_config["post_processing_strategy"] == "fill_first":
                            is_pixel_without_prediction = torch.isnan(y_hat[:, :, i, j]).any()
                            if is_pixel_without_prediction:
                                y_hat[:, :, i, j] = patch_prediction[:, :, i - patch_coordinates[0], j - patch_coordinates[1]]
                        elif self.training_config["post_processing_strategy"] == "weighted_average":
                            y_hat_weighted_avg_method[:, :, i, j] += patch_prediction[:, :, i - patch_coordinates[0], j - patch_coordinates[1]]
                            counter_grid[i, j] += 1
                            y_hat[:, :, i, j] = y_hat_weighted_avg_method[:, :, i, j] / counter_grid[i, j]
                        else:
                            raise ValueError(
                                f"Invalid configuration value: {self.training_config['post_processing_strategy']}. "
                                f"Allowed values are: ['fill_first', 'weighted_average']"
                            )

            ##############  POST-PROCESSING   ###############
            
            # Apply softmax to probabilities
            y_hat_probabilities = F.softmax(y_hat, dim=1)

            ##############  LOSS   ###############
            
            # Compute loss and store it in the list
            loss = self.loss_fn(y_hat_probabilities, y)
            losses_batch.append(loss.item())

            ############## (METRICS) POST-PROCESSING   ###############

            # Convert predictions to one-hot encoding
            y_hat_one_hot = torch.zeros_like(y_hat_probabilities)
            max_indices = torch.argmax(y_hat_probabilities, dim=1)
            y_hat_one_hot.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)

            ##############  MEAN IOU METRIC   ###############

            # Compute Mean IoU 
            iou = self.mean_iou_metric(y_pred=y_hat_one_hot, y=y)  # Directly call MeanIoU
            mean_iou = iou.mean()  # Mean IoU across classes
            mean_ious_batch.append(mean_iou.item())

            ##############  DICE COEFF METRIC   ###############

            # Compute Dice Metric (Dice Coeff) 
            dice = self.dice_metric(y_pred=y_hat_one_hot, y=y)  # Directly call DiceMetric
            mean_dice = dice.mean()  # Average Dice across classes
            dice_coeffs_batch.append(mean_dice.item())

            ##############  FPR METRIC   ###############

            # Compute FPR Metric
            # Step 1: Classify each pixel accoring to probabilties assigning the class with higher probabilty and flatten it
            y_hat_categorized_flattened = torch.argmax(y_hat_probabilities, dim=1).view(-1)  # Shape: (batch_size * height * width)
            # Step 2: Convert y one_encoded into categorized tensor and flatten it
            y_categorized_flattened = torch.argmax(y, dim=1).view(-1)  # Shape: (batch_size * height * width)
            # Step 3: Define the number of classes
            num_classes = y_hat_probabilities.shape[1]
            # Step 4: Compute the confusion matrix using torchmetrics
            confmat = MulticlassConfusionMatrix(num_classes=y_hat_probabilities.shape[1]).to("cuda")
            confusion_matrix = confmat(y_hat_categorized_flattened, y_categorized_flattened)
            # Step 5: Calculate False Positives (FP) and True Negatives (TN)
            tp = torch.diag(confusion_matrix)
            fp = confusion_matrix.sum(dim=0) - tp
            fn = confusion_matrix.sum(dim=1) - tp
            tn = confusion_matrix.sum() - (tp + fp + fn)
            # Step 6: Calculate FPR for each class
            fpr = fp / (fp + tn)
            tpr_recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            # Step 7: Exclude the background class (class 0) (if activated)
            if not self.include_background_in_loss_and_metrics:
                fpr = fpr[1:]
                tpr_recall = tpr_recall[1:]
                precision = precision[1:]
            avg_fpr_classes = fpr.mean()
            fprs_batch.append(avg_fpr_classes.item())
            avg_tpr_recall_classes = tpr_recall.mean()
            tpr_recalls_batch.append(avg_tpr_recall_classes.item())
            avg_precision_classes = precision.mean()
            precisions_batch.append(avg_precision_classes.item())

            ############## (FINAL OUTPUT SEGMENTATION) POST-PROCESSING   ###############

            # Classify each pixel accoring to probabilties assigning the class with higher probabilty
            y_hat_categorized = torch.argmax(y_hat_probabilities.squeeze(), dim=0)
            # Convert y one_encoded into categorized tensor
            y_categorized = torch.argmax(y.squeeze(), dim=0)
    
            # Save categorized ground_truth and prediction
            self.test_sample_ids.append(sample_id)
            self.test_ground_truths.append(y_categorized)
            self.test_predictions.append(y_hat_categorized)
            self.test_predictions_probabilities.append(y_hat_probabilities.squeeze()[1]) # Probilities positive class (non-background)

            

        #############################   LOSS   #############################################
        
        # Calculate average loss for the batch
        #batch_loss_avg = torch.tensor(losses_batch, device=self.device).mean()
        batch_loss_avg = np.mean(losses_batch)
        self.log('test_loss', batch_loss_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))

        if self.enable_batch_logging_into_console and self.enable_testing_epoch_logging_into_consolfe:
            print(f"- Dice Loss: {batch_loss_avg:.6f}")

        self.testing_batch_losses_in_epoch.append(batch_loss_avg)

        ############################   METRICS   ############################################

        # Calculate average dice_coeff for the batch
        batch_mean_iou_avg = np.mean(mean_ious_batch)
        self.log('test_mean_iou', batch_mean_iou_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.testing_batch_mean_ious_in_epoch.append(batch_mean_iou_avg)        
        

        # Calculate average loss for the batch
        batch_dice_avg = np.mean(dice_coeffs_batch)
        self.log('test_dice', batch_dice_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.testing_batch_dice_coeffs_in_epoch.append(dice_coeffs_batch)

        batch_fpr_avg = np.mean(fprs_batch)
        self.log('test_fpr', batch_fpr_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.testing_batch_fprs_in_epoch.append(batch_fpr_avg)

        batch_tpr_recall_avg = np.mean(tpr_recalls_batch)
        self.log('test_tpr_recall', batch_tpr_recall_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.testing_batch_tpr_recalls_in_epoch.append(batch_tpr_recall_avg)

        batch_precision_avg = np.mean(precisions_batch)
        self.log('test_precision', batch_precision_avg, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.testing_batch_precisions_in_epoch.append(batch_precision_avg)

        # Return the average batch loss
        return batch_loss_avg

    def configure_optimizers(self):        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config["lr_optimizer"], weight_decay=self.training_config["weight_decay_optimizer"])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs = self.training_config["warmup_epochs_lr_scheduler"], max_epochs=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    # Overriding hooks for demonstration
    def on_train_epoch_start(self):

        if self.enable_training_epoch_logging_into_console:
            # Start of an epoch
            print(f"{'=' * 40}")
            print(f"Starting training epoch {self.current_epoch}...")
            print(f"{'=' * 40}")

    def on_train_epoch_end(self):
        
        # Average Dice Loss for the Epoch
        avg_loss = np.mean(self.training_batch_losses_in_epoch)
        #average_loss_for_epoch = torch.stack(self.training_batch_losses_in_epoch).mean()
        # Clearing list for the next epoch
        self.training_batch_losses_in_epoch.clear()
        
        if self.enable_training_epoch_logging_into_console:
            print()
            print(f"==> (TRAINING) Average Dice Loss (Patch Level) (include_background=[{self.include_background_in_loss_and_metrics}]): {avg_loss:.6f}")
            print()

            # End of an epoch
            print(f"{'=' * 40}")
            print(f"Finished training epoch {self.current_epoch}")
            print(f"{'=' * 40}")
            print()

    def on_validation_epoch_start(self):
        self.val_ground_truths = []
        self.val_predictions = []
        self.val_sample_ids = []


    def on_validation_epoch_end(self):
        
        avg_loss = np.mean(self.validation_batch_losses_in_epoch)
        self.validation_batch_losses_in_epoch.clear()

        if self.best_epoch_val_loss > avg_loss:
            self.best_epoch_val_loss = avg_loss
            self.best_epoch_val_ground_truths = self.val_ground_truths
            self.best_epoch_val_predictions = self.val_predictions
            self.best_epoch_val_sample_ids = self.val_sample_ids

        average_mean_iou_epoch = np.mean(self.validation_batch_mean_ious_in_epoch)
        self.validation_batch_mean_ious_in_epoch.clear()

        average_dice_coeff_epoch = np.mean(self.validation_batch_dice_coeffs_in_epoch)
        self.validation_batch_dice_coeffs_in_epoch.clear()

        average_fpr_epoch = np.mean(self.validation_batch_fprs_in_epoch)
        self.validation_batch_fprs_in_epoch.clear()
        
        if self.enable_validation_epoch_logging_into_console:
            # print(f"{'=' * 40}")
            # print(f"Finished validation epoch")
            # print(f"{'=' * 40}")

            print()
            print(f"==> (VALIDATION) Average Dice Loss (include_background=[{self.include_background_in_loss_and_metrics}]): {avg_loss:.6f}")
            print()

            print(f"==> (VALIDATION) Average Mean IoU (include_background=[{self.include_background_in_loss_and_metrics}]): {average_mean_iou_epoch:.6f}")
            print()

            print(f"==> (VALIDATION) Average Dice Coefficient (include_background=[{self.include_background_in_loss_and_metrics}]): {average_dice_coeff_epoch:.6f}")
            print()

            print(f"==> (VALIDATION) Average FPR (include_background=[{self.include_background_in_loss_and_metrics}]): {average_fpr_epoch:.6f}")
            print()

    def on_test_epoch_start(self):
        self.test_ground_truths = []
        self.test_predictions = []
        self.test_sample_ids = []
        self.test_predictions_probabilities = []

        if self.enable_testing_epoch_logging_into_console:
                # Start of an epoch
                print(f"{'=' * 40}")
                print(f"Starting testing epoch...")
                print(f"{'=' * 40}")


    def on_test_epoch_end(self):
        
        self.test_loss = np.mean(self.testing_batch_losses_in_epoch)
        self.testing_batch_losses_in_epoch.clear()

        self.test_mean_iou = np.mean(self.testing_batch_mean_ious_in_epoch)
        self.testing_batch_mean_ious_in_epoch.clear()

        self.test_dice_coeff = np.mean(self.testing_batch_dice_coeffs_in_epoch)
        self.testing_batch_dice_coeffs_in_epoch.clear()

        self.test_fpr = np.mean(self.testing_batch_fprs_in_epoch)
        self.testing_batch_fprs_in_epoch.clear()

        self.test_tpr_recall = np.mean(self.testing_batch_tpr_recalls_in_epoch)
        self.testing_batch_tpr_recalls_in_epoch.clear()

        self.test_precision = np.mean(self.testing_batch_precisions_in_epoch)
        self.testing_batch_precisions_in_epoch.clear()
        
        if self.enable_testing_epoch_logging_into_console:

            print()
            print(f"==> (TESTING) Average Dice Loss (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_loss:.6f}")
            print()   
            
            print(f"==> (TESTING) Average Mean IoU (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_mean_iou:.6f}")
            print()
            
            print(f"==> (TESTING) Average Dice Coefficient (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_dice_coeff:.6f}")
            print()

            print(f"==> (TESTING) Average FPR (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_fpr:.6f}")
            print()

            print(f"==> (TESTING) Average TPR/Recall (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_tpr_recall:.6f}")
            print()

            print(f"==> (TESTING) Average Precision (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_precision:.6f}")
            print()
            

            print(f"{'=' * 40}")
            print(f"Finished testing epoch")
            print(f"{'=' * 40}")
            

