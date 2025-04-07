# PyTorch and PyTorch Lightning
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from scipy.ndimage import label
from skimage.measure import regionprops
import numpy as np
from sklearn.metrics import roc_curve, auc  # For ROC curve computation and AUC calculation
from scipy.integrate import simpson  # For numerical integration (Simpson's Rule)


# Numpy
import numpy as np

# Optimizer
from torch.optim import AdamW
#from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
from models.unet import UNET_VGG11_architecture
importlib.reload(UNET_VGG11_architecture)
from models.unet.UNET_VGG11_architecture import UNet_VGG11

print("All imports succeeded!")

# Define a PyTorch Lightning model wrapper
class UNET_VGG11_LightningModel(pl.LightningModule):
    def __init__(self, lr_optimizer, weight_decay_optimizer, optimizer, **kwargs):
        super(UNET_VGG11_LightningModel, self).__init__()

        ###################### MODEL INSTANTIATION ###########################

        self.model = UNet_VGG11(in_channels=10, out_classes=2, **kwargs).to("cuda")

        ################# OPTIMIZER/LR_SCHEDULER PARAMETERS #################

        self.lr = lr_optimizer
        self.weight_decay = weight_decay_optimizer
        self.optimizer = optimizer
        
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

        
        self.test_loss = None
        self.test_mean_iou = None
        self.test_dice_coeff = None
        self.test_fpr = None
        self.test_tpr_recall = None
        self.test_precision = None
        # self.test_defectwise_iou_scores= None
        self.test_ground_truths = []
        self.test_predictions = []
        self.test_sample_ids = []
    
        self.val_epoch_loss = None
        self.val_epoch_mean_iou = None
        self.val_epoch_dice_coeff = None
        self.val_epoch_fpr = None
        # self.val_epoch_defectwise_iou_scores = []
        self.val_epoch_ground_truths = []
        self.val_epoch_predictions = []
        self.val_epoch_sample_ids = []

        self.best_epoch_val = None
        self.best_epoch_val_loss = float('inf')
        self.best_epoch_val_mean_iou= None
        self.best_epoch_val_dice_coeff = None
        self.best_epoch_val_fpr = None
        # self.best_epoch_val_defectwise_iou_scores = []
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
        sample_ids, x, y = batch  # sample_ids shape: (bacth_size, 1), x shape: (batch_size, input_channels, height, width), y shape: (batch_size, output_channels, height, width)
    
        # Forward pass through the model
        y_hat = self(x)  # y_hat shape: (batch_size, output_channels, height, width)
    
        # Apply post-processing (e.g., softmax for probabilities)
        y_hat_probabilities = F.softmax(y_hat, dim=1)
    
        # Compute loss using the provided loss function
        loss = self.loss_fn(y_hat_probabilities, y)

        # Keep track of batch losses for custom epoch-level logging
        self.training_batch_losses_in_epoch.append(loss.item())
    
        # Log training loss
        self.log(
            'train_loss', 
            loss.item(), 
            on_step=True, 
            on_epoch=True, 
            prog_bar=False, 
            batch_size=self.trainer.train_dataloader.batch_size
        )
    
        return loss  # Return loss for Lightning to handle the backward pass
        
    
    def validation_step(self, batch, batch_idx):
        
        if self.enable_batch_logging_into_console and self.enable_validation_epoch_logging_into_console:
            print(f"Validation: Epoch {self.current_epoch}, Batch {batch_idx}")

        sample_ids, x, y = batch  # sample_ids shape: (bacth_size, 1), x shape: (batch_size, input_channels, height, width), y shape: (batch_size, output_channels, height, width)

        # Forward pass through the model
        y_hat = self(x)  # y_hat shape: (batch_size, output_channels, height, width)        

        ##############  POST-PROCESSING   ###############
            
        # Apply softmax to probabilities
        y_hat_probabilities = F.softmax(y_hat, dim=1)

        ##############  LOSS   ###############
        
        loss = self.loss_fn(y_hat_probabilities, y) # Returns a tensor representing the average Dice loss over the batch e.g (0.3452, grad_fn=<MeanBackward0>) where 0.3452 is the average Dice loss
        self.validation_batch_losses_in_epoch.append(loss.item())
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])

        if self.enable_batch_logging_into_console and self.enable_validation_epoch_logging_into_console:
            print(f"- Average Dice Loss (Batch) (include_background=[{self.include_background_in_loss_and_metrics}]): {loss.item():.6f}")

        ############## (METRICS) POST-PROCESSING   ###############
        
        ################################################
        
        # Convert predictions to one-hot encoding
        y_hat_one_hot = torch.zeros_like(y_hat_probabilities)
        max_indices = torch.argmax(y_hat_probabilities, dim=1)
        y_hat_one_hot.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)

        ##############  MEAN IOU METRIC   ###############

        # Compute Mean IoU 
        iou = self.mean_iou_metric(y_pred=y_hat_one_hot, y=y)  # Directly call MeanIoU
        mean_iou = iou.mean().item()  # Mean IoU across all classes and all samples in the batch. 
        self.log('val_mean_iou', mean_iou, on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])
        self.validation_batch_mean_ious_in_epoch.append(mean_iou) 

        ##############  DICE COEFF METRIC   ###############

        # Compute Dice Metric (Dice Coeff) 
        dice = self.dice_metric(y_pred=y_hat_one_hot, y=y)  # Directly call DiceMetric
        mean_dice = dice.mean().item()  # Average Dice across all classes and all samples in the batch.
        self.log('val_dice', mean_dice, on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])
        self.validation_batch_dice_coeffs_in_epoch.append(mean_dice)

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
        avg_fpr_classes = fpr.mean().item()
        self.log('val_fpr', avg_fpr_classes, on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])
        self.validation_batch_fprs_in_epoch.append(avg_fpr_classes)

        ############## (FINAL OUTPUT SEGMENTATION) POST-PROCESSING  + DEFECT-WISE METRICS ###############

        for i in range(0, x.shape[0]):
            # Classify each pixel accoring to probabilties assigning the class with higher probabilty
            y_hat_categorized = torch.argmax(y_hat_probabilities[i], dim=0)
            # Convert y one_encoded into categorized tensor
            y_categorized = torch.argmax(y[i], dim=0)

            # Save categorized ground_truth and prediction
            self.val_epoch_ground_truths.append(y_categorized)
            self.val_epoch_predictions.append(y_hat_categorized)
            self.val_epoch_sample_ids.append(sample_ids[i])
               
        # Return loss
        return loss



    def test_step(self, batch, batch_idx):
        
        if self.enable_batch_logging_into_console and self.enable_testing_epoch_logging_into_console:
            print(f"Validation: Epoch {self.current_epoch}, Batch {batch_idx}")

        sample_ids, x, y = batch  # sample_ids shape: (bacth_size, 1), x shape: (batch_size, input_channels, height, width), y shape: (batch_size, output_channels, height, width)

        # Forward pass through the model
        y_hat = self(x)  # y_hat shape: (batch_size, output_channels, height, width)

        ##############  POST-PROCESSING   ###############
        
        # Apply softmax to probabilities
        y_hat_probabilities = F.softmax(y_hat, dim=1)

        ##############  LOSS   ###############
        
        # Compute loss and store it in the list
        loss = self.loss_fn(y_hat_probabilities, y)
        self.log('test_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])
        self.testing_batch_losses_in_epoch.append(loss.item())
        
        if self.enable_batch_logging_into_console and self.enable_testing_epoch_logging_into_consolfe:
            print(f"- Dice Loss: {loss.item():.6f}")

        ############## (METRICS) POST-PROCESSING   ###############

        # Convert predictions to one-hot encoding
        y_hat_one_hot = torch.zeros_like(y_hat_probabilities)
        max_indices = torch.argmax(y_hat_probabilities, dim=1)
        y_hat_one_hot.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)

        ##############  MEAN IOU METRIC   ###############

        # Compute Mean IoU 
        iou = self.mean_iou_metric(y_pred=y_hat_one_hot, y=y)  # Directly call MeanIoU
        mean_iou = iou.mean().item()  # Mean IoU across all classes and all samples in the batch. iou.mean() returns a scalar
        self.testing_batch_mean_ious_in_epoch.append(mean_iou)
        self.log('test_mean_iou', mean_iou, on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])

        ##############  DICE COEFF METRIC   ###############

        # Compute Dice Metric (Dice Coeff) 
        dice = self.dice_metric(y_pred=y_hat_one_hot, y=y)  # Directly call DiceMetric
        mean_dice = dice.mean().item()  # Average Dice across all classes and all samples in the batch. Return a scalar
        self.testing_batch_dice_coeffs_in_epoch.append(mean_dice)  
        self.log('test_dice', mean_dice, on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])

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
            
        avg_fpr_classes = fpr.mean().item()
        self.log('test_fpr', avg_fpr_classes, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.testing_batch_fprs_in_epoch.append(avg_fpr_classes)

        avg_tpr_recall_classes = tpr_recall.mean().item()
        self.log('test_tpr_recall', avg_tpr_recall_classes, on_step=True, on_epoch=True, prog_bar=False, batch_size=x.shape[0])
        self.testing_batch_tpr_recalls_in_epoch.append(avg_tpr_recall_classes)
        
        avg_precision_classes = precision.mean().item()
        self.log('test_precision', avg_precision_classes, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(x))
        self.testing_batch_precisions_in_epoch.append(avg_precision_classes)

        ############## (FINAL OUTPUT SEGMENTATION) POST-PROCESSING + DEEFCT-WISE METRICS  ###############

        for i in range(0, x.shape[0]):
            # Classify each pixel accoring to probabilties assigning the class with higher probabilty
            y_hat_categorized = torch.argmax(y_hat_probabilities[i], dim=0)
            # Convert y one_encoded into categorized tensor
            y_categorized = torch.argmax(y[i], dim=0)
    
            # Save categorized ground_truth and prediction
            self.test_ground_truths.append(y_categorized)
            self.test_predictions.append(y_hat_categorized)
            self.test_sample_ids.append(sample_ids[i])
        
        # Return the loss
        return loss.item()

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        elif  self.optimizer == "sgdm":    
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"❌ Invalid optimizer '{self.optimizer}'. Choose from ['adam', 'sgdm'].")

        return optimizer

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
        self.val_epoch_loss = None
        self.val_epoch_mean_iou = None
        self.val_epoch_dice_coeff = None
        self.val_epoch_fpr = None
        self.val_epoch_ground_truths = []
        self.val_epoch_predictions = []
        self.val_epoch_sample_ids = []


    def on_validation_epoch_end(self):
        self.val_epoch_loss = np.mean(self.validation_batch_losses_in_epoch)
        self.validation_batch_losses_in_epoch.clear()

        self.val_epoch_mean_iou = np.mean(self.validation_batch_mean_ious_in_epoch)
        self.validation_batch_mean_ious_in_epoch.clear()

        self.val_epoch_dice_coeff = np.mean(self.validation_batch_dice_coeffs_in_epoch)
        self.validation_batch_dice_coeffs_in_epoch.clear()

        self.val_epoch_fpr = np.mean(self.validation_batch_fprs_in_epoch)
        self.validation_batch_fprs_in_epoch.clear()

        if self.best_epoch_val_loss > self.val_epoch_loss:

            print()
            print("Best epoch updated")
            
            self.best_epoch_val = self.current_epoch
            self.best_epoch_val_loss = self.val_epoch_loss
            self.best_epoch_val_mean_iou=  self.val_epoch_mean_iou
            self.best_epoch_val_dice_coeff = self.val_epoch_dice_coeff
            self.best_epoch_val_fpr = self.val_epoch_fpr
            self.best_epoch_val_ground_truths = self.val_epoch_ground_truths
            self.best_epoch_val_predictions = self.val_epoch_predictions
            self.best_epoch_val_sample_ids = self.val_epoch_sample_ids
            
        
        if self.enable_validation_epoch_logging_into_console:

            print()
            print(f"==> (VALIDATION) Average Dice Loss (include_background=[{self.include_background_in_loss_and_metrics}]): {self.val_epoch_loss:.6f}")
            print()

            print(f"==> (VALIDATION) Average Pixel-Wise Mean IoU (include_background=[{self.include_background_in_loss_and_metrics}]): {self.val_epoch_mean_iou:.6f}")
            print()

            print(f"==> (VALIDATION) Average Pixel-Wise Dice Coefficient (include_background=[{self.include_background_in_loss_and_metrics}]): {self.val_epoch_dice_coeff:.6f}")
            print()

            print(f"==> (VALIDATION) Average Pixel-Wise FPR (include_background=[{self.include_background_in_loss_and_metrics}]): {self.val_epoch_fpr:.6f}")
            print()


    def on_test_epoch_start(self):
        self.test_loss = None
        self.test_mean_iou = None
        self.test_dice_coeff = None
        self.test_fpr = None
        self.test_tpr_recall = None
        self.test_precision = None
        self.test_ground_truths = []
        self.test_predictions = []
        self.test_sample_ids = []

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
            
            print(f"==> (TESTING) Average Pixel-Wise Mean IoU (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_mean_iou:.6f}")
            print()
            
            print(f"==> (TESTING) Average Pixel-Wise Dice Coefficient (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_dice_coeff:.6f}")
            print()

            print(f"==> (TESTING) Average Pixel-Wise FPR (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_fpr:.6f}")
            print()

            print(f"==> (TESTING) Average TPR/Recall (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_tpr_recall:.6f}")
            print()

            print(f"==> (TESTING) Average Precision (include_background=[{self.include_background_in_loss_and_metrics}]): {self.test_precision:.6f}")
            print()

            print(f"{'=' * 40}")
            print(f"Finished testing epoch")
            print(f"{'=' * 40}")
            

