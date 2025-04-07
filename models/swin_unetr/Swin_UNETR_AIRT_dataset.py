import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from itertools import product
from scipy.ndimage import gaussian_filter
import scipy.io as io
import shutil
from datetime import datetime
import time

# Define a custom 3D dataset for testing purposes
class SwinUNETR_AIRT_Dataset(Dataset):
    def __init__(self, is_inference_mode: bool, augmentation: bool,  metadata_dict_with_files_selected: dict, data_dir: str, model_input_dims: tuple, overlap , include_plotting: bool = False, preprocessing_strategy: str  = "uniform_gaussian_smoothed_equidistant_subsampling"):
        """
        Initializes the class instance with the provided parameters.

        Parameters:
        metadata_dict_with_files_selected (dict): Dictionary containing the selected files for the dataset with their metadata (e.g., 3D_thermal_sequence_filename, label_filename, ROI, stratified group, etc.).
        data_dir (str): Directory where the data is located.
        model_input_dim (tuple): 3D tuple indicating the dimensions of the model input (height, width, depth).
        """

        self.metadata_dict_with_files_selected = metadata_dict_with_files_selected
        self.data_dir = data_dir

        self.is_inference_mode = is_inference_mode # e.g. False

        self.augmentation = augmentation

        self.include_plotting = include_plotting

        self.model_input_dim = model_input_dims #e.g. (64,64,128)

        self.preprocessing_mode = preprocessing_strategy

        self.overlap = overlap #e.g. [0.25,0.35]
        formatted_overlap_dim_0 = f"{self.overlap[0]:.2f}"
        formatted_overlap_dim_1 = f"{self.overlap[1]:.2f}"

        self.patch_size = f"{self.model_input_dim[0]}x{self.model_input_dim[1]}" #e.g. 64x64
        self.overlap_key = f"{formatted_overlap_dim_0.replace('.', '_')}x{formatted_overlap_dim_1.replace('.', '_')}" #e.g. 0_25x0_35

        self.preprocessed_dir = os.path.join(os.getcwd(), self.data_dir, "preprocessed_files")

        self.preprocessed_patches_dataset = []

        # ########################### DATA AUGMENTATION (AND DISK SAVING) ###########################

        if self.augmentation:
            print(f"==> INCLUDING IN DATASET AUGMENTED VERSIONS FROM ORIGINAL SELECTED VIDEOS")
            print()
            print(f"Original selected videos: {list(metadata_dict_with_files_selected.keys())}")
            print()
            
            new_entries = {}
            
            for sample_id, sample_metadata in self.metadata_dict_with_files_selected.items():
                print(f"- Sample: {sample_id}")
                if "augmentations" in self.metadata_dict_with_files_selected[sample_id].keys():
                    for augmentation in self.metadata_dict_with_files_selected[sample_id]["augmentations"]:
                        aug_key, aug_data = list(augmentation.items())[0]
                        
                        # Store in temporary dictionary
                        new_entries[aug_key] = aug_data
                        print(f"- Augmented Video {aug_key} Included")

            # Update the dictionary outside the loop
            self.metadata_dict_with_files_selected.update(new_entries)
            print()

        # ########################### PREPROCESSING (AND DISK SAVING) ###########################

        print(f"==> DATA PREPOCESSING STARTS")
        print()
        start_time = datetime.now()
        print("Data preprocessing started at:", start_time)
        print()


        if not os.path.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            print(f"The directory '{self.preprocessed_dir}' did not exist and has been created.")
            print()

        self.preprocessed_dir = os.path.join(self.preprocessed_dir, "swin_unetr")
        if not os.path.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            print(f"The directory '{self.preprocessed_dir}' did not exist and has been created.")
            print()

                ####### NEW CODE FOR REMOVING PREPROCCESED FILES FOR EVERY DATASET INSTANTIATION #########

        if self.is_inference_mode:
            self.preprocessed_dir = os.path.join(self.preprocessed_dir, "inference")
        else:
            self.preprocessed_dir = os.path.join(self.preprocessed_dir, "training")
            
        if not os.path.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            print(f"The directory '{self.preprocessed_dir}' did not exist and has been created.")
            print()
        else:
            print(f"Removing the directory '{self.preprocessed_dir}' and its content ...")
            shutil.rmtree(self.preprocessed_dir)
            print(f"The directory '{self.preprocessed_dir}' and its content have been removed.")
            os.makedirs(self.preprocessed_dir)
            print(f"The directory '{self.preprocessed_dir}' did not exist and has been created.")
            print()

                ############################### NEW CODE FINISHED ########################################

        self.preprocessed_info_json_path = os.path.join(self.preprocessed_dir, "preprocessed_info.json")
        self.preprocessed_info_dict = None

        if os.path.isfile(self.preprocessed_info_json_path):
            self.preprocessed_info_dict = self.load_preprocessed_info()
        else:
            self.preprocessed_info_dict = {}

        for sample_id, sample_metadata in self.metadata_dict_with_files_selected.items():

            print(f"Preprocessing sample {sample_id} ...")
            print(f"{'=' * 40}")
            print()

            start_sample_time = time.time()  # ⏱️ Iniciar contador
            
            # Checks
            if not sample_id in self.preprocessed_info_dict:
                # Create directory for the sample
                preprocessed_sample_dir = os.path.join(self.preprocessed_dir, f"{sample_id}")
                os.makedirs(preprocessed_sample_dir)
                #print(f"The directory '{preprocessed_sample_dir}' did not exist and has been created.")
                preprocessed_sample_label_dir = os.path.join(preprocessed_sample_dir, "label")
                os.makedirs(preprocessed_sample_label_dir)
                #print(f"The directory '{preprocessed_sample_label_dir}' did not exist and has been created.")
                preprocessed_sample_label_patches_dir = os.path.join(preprocessed_sample_dir, "label", "patches")
                os.makedirs(preprocessed_sample_label_patches_dir)
                #print(f"The directory '{preprocessed_sample_label_patches_dir}' did not exist and has been created.")
                preprocessed_sample_measurement_dir = os.path.join(preprocessed_sample_dir, "measurement")
                os.makedirs(preprocessed_sample_measurement_dir)
                #print(f"The directory '{preprocessed_sample_measurement_dir}' did not exist and has been created.")
                
                # Update the information in the preprocessed_info_json
                self.update_dictionary(self.preprocessed_info_dict, sample_id, {})
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement" ,{})
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "label" ,{})
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "label" ,"patches", {})

            #Check depth directory for measurement depth
            if not f"depth[{self.model_input_dim[2]}]" in self.preprocessed_info_dict[sample_id]["measurement"]:
                # Create directory for the depth we will apply inside the measurement directory
                measurement_depth_dir = os.path.join(self.preprocessed_dir, f"{sample_id}", "measurement", f"depth[{self.model_input_dim[2]}]")
                os.makedirs(measurement_depth_dir)
                #print(f"The directory '{measurement_depth_dir}' did not exist and has been created.")
                # Update the information in the preprocessed_info_json
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", {})

            #Check for preprocessing mode
            if not f"preprocessing_mode[{self.preprocessing_mode}]" in self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"]:
                # Create directory for the depth we will apply inside the measurement directory
                preprocessing_mode_dir = os.path.join(self.preprocessed_dir, f"{sample_id}", "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]")
                os.makedirs(preprocessing_mode_dir)
                #print(f"The directory '{preprocessing_mode_dir}' did not exist and has been created.")
                measurement_depth_patches_dir = os.path.join(preprocessing_mode_dir, "patches")
                os.makedirs(measurement_depth_patches_dir)
                #print(f"The directory '{measurement_depth_patches_dir}' did not exist and has been created.")
                # Update the information in the preprocessed_info_json
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", {})
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", "patches", {})

            #Check patches_size directory for label
            if not f"patch_size[{self.patch_size}]" in self.preprocessed_info_dict[sample_id]["label"]["patches"]:
                label_patches_size_dir = os.path.join(self.preprocessed_dir, f"{sample_id}", "label", "patches", f"patch_size[{self.patch_size}]")
                os.makedirs(label_patches_size_dir)
                #print(f"The directory '{label_patches_size_dir}' did not exist and has been created.")
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "label", "patches", f"patch_size[{self.patch_size}]", {})
                # Create and save the cropped label in the directory just created
                cropped_label_tensor, cropped_bbox_info = self.load_and_crop_label(sample_id, sample_metadata)
                cropped_label_tensor_filename = os.path.join(label_patches_size_dir,f"{sample_id}_cropped[{cropped_bbox_info['height']}x{cropped_bbox_info['width']}]_label.raw")
                # Convert the tensor to numpy.array so the patch values can be stored in raw binary format with .tofile()
                cropped_label_tensor_array = cropped_label_tensor.numpy()
                cropped_label_tensor_array.tofile(cropped_label_tensor_filename)
                #print(f"File was generated and saved successfully: {cropped_label_tensor_filename}")             
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "label", "patches", f"patch_size[{self.patch_size}]", "cropped_label_tensor_filename", cropped_label_tensor_filename)
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "label", "patches", f"patch_size[{self.patch_size}]", "cropped_label_tensor_shape", list(cropped_label_tensor.shape))
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "label", "patches", f"patch_size[{self.patch_size}]", "cropped_bbox_info", cropped_bbox_info)

            #Check patches_size directory for measurement
            if not f"patch_size[{self.patch_size}]" in self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"][f"preprocessing_mode[{self.preprocessing_mode}]"]["patches"]:
                measurement_depth_patches_size_dir = os.path.join(self.preprocessed_dir, f"{sample_id}", "measurement", f"depth[{self.model_input_dim[2]}]",f"preprocessing_mode[{self.preprocessing_mode}]", "patches", f"patch_size[{self.patch_size}]")
                os.makedirs(measurement_depth_patches_size_dir)
                #print(f"The directory '{measurement_depth_patches_size_dir}' did not exist and has been created.")
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", "patches", f"patch_size[{self.patch_size}]", {})
                 # Create and save the cropped, standarized & depth compressed 3D measurement in the directory just created
                cropped_standarized_depth_compressed_measurement_tensor, cropped_bbox_info = self.load_crop_standardize_depth_compress_measurement(sample_id, sample_metadata)
                cropped_standarized_depth_compressed_measurement_tensor_filename = os.path.join(measurement_depth_patches_size_dir, f"{sample_id}_cropped[{cropped_bbox_info['height']}x{cropped_bbox_info['width']}]_standarized_depth[{self.model_input_dim[2]}]_preprocessing_mode[{self.preprocessing_mode}.raw")
                # Convert the tensor to numpy.array so the patch values can be stored in raw binary format with .tofile()
                cropped_standarized_depth_compressed_measurement_tensor_array = cropped_standarized_depth_compressed_measurement_tensor.numpy()
                cropped_standarized_depth_compressed_measurement_tensor_array.tofile(cropped_standarized_depth_compressed_measurement_tensor_filename)
                #print(f"File was generated and saved successfully: {cropped_standarized_depth_compressed_measurement_tensor_filename}")
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", "patches", f"patch_size[{self.patch_size}]", "cropped_standarized_depth_compressed_measurement_tensor_filename", cropped_standarized_depth_compressed_measurement_tensor_filename)
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", "patches", f"patch_size[{self.patch_size}]", "cropped_standarized_depth_compressed_measurement_tensor_shape", list(cropped_standarized_depth_compressed_measurement_tensor.shape))
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", "patches", f"patch_size[{self.patch_size}]", "cropped_bbox_info", cropped_bbox_info)


            #LABEL_PATCHES
            if f"overlap[{self.overlap_key}]" in self.preprocessed_info_dict[sample_id]["label"]["patches"][f"patch_size[{self.patch_size}]"]:
                pass
                #print(f"The label corresponding to '{sample_id}' sample has already been processed with patch size [{self.patch_size}] and overlap [{self.overlap_key}]")
            else:
                label_overlap_dir = os.path.join(self.preprocessed_dir, f"{sample_id}", "label", "patches", f"patch_size[{self.patch_size}]", f"overlap[{self.overlap_key}]")
                os.makedirs(label_overlap_dir)
                #print(f"The directory '{label_overlap_dir}' did not exist and has been created.")
                cropped_bbox_info = self.preprocessed_info_dict[sample_id]["label"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_bbox_info"]
                cropped_label_tensor_filename = self.preprocessed_info_dict[sample_id]["label"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_label_tensor_filename"]

                ##############  LOADING cropped_label_tensor   ###############
                cropped_label_tensor_shape = tuple(self.preprocessed_info_dict[sample_id]["label"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_label_tensor_shape"])
                # Load the raw data into a NumPy array
                loaded_array_cropped_label = np.fromfile(cropped_label_tensor_filename, dtype=np.float32)           
                # Reshape the loaded array to the original shape
                loaded_array_cropped_label = loaded_array_cropped_label.reshape(cropped_label_tensor_shape)           
                # Convert back to a PyTorch tensor
                cropped_label_tensor = torch.from_numpy(loaded_array_cropped_label)
                ##########################################
                
                #cropped_label_tensor = torch.load(cropped_label_tensor_filename, weights_only=True)
                label_patches = self.create_patches(cropped_label_tensor, sample_metadata["ROI"], cropped_bbox_info)

                label_patches_dictionary = {}
                # Save patches
                for i, label_patch in enumerate(label_patches):
                    label_tensor_patch, label_tensor_patch_coord, roi_adjusted_patch = label_patch
                    preprocessed_label_patch_filename = os.path.join(f"label_{sample_id}_patch_size[{self.patch_size}]_overlap[{self.overlap_key}]_coords[{label_tensor_patch_coord[0]}x{label_tensor_patch_coord[1]}].raw")
                    patch_path= os.path.join(label_overlap_dir, preprocessed_label_patch_filename)
                    # Convert the tensor to numpy.array so the patch values can be stored in raw binary format with .tofile()
                    label_tensor_patch_array = label_tensor_patch.numpy()
                    label_tensor_patch_array.tofile(patch_path)
                    #torch.save(label_tensor_patch, patch_path)
                    label_patches_dictionary["_".join(map(str, label_tensor_patch_coord))] = {
                        "patch_coord": label_tensor_patch_coord,
                        "patch_path": patch_path,
                        "patch_tensor_shape": list(label_tensor_patch.shape),
                        "roi_adjusted_patch": roi_adjusted_patch
                    }
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "label", "patches", f"patch_size[{self.patch_size}]", f"overlap[{self.overlap_key}]", label_patches_dictionary)

            #MEASUREMENT_PATCHES
            if f"overlap[{self.overlap_key}]" in self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"][f"preprocessing_mode[{self.preprocessing_mode}]"]["patches"][f"patch_size[{self.patch_size}]"]:
                pass
                #print(f"The measurement corresponding to '{sample_id}' sample has already been processed with depth [{self.model_input_dim[2]}], preprocessing_mode [{self.preprocessing_mode}], patch size [{self.patch_size}] and overlap [{self.overlap_key}]")
            else:
                measurement_overlap_dir = os.path.join(self.preprocessed_dir, f"{sample_id}", "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", "patches", f"patch_size[{self.patch_size}]", f"overlap[{self.overlap_key}]")
                os.makedirs(measurement_overlap_dir)
                #print(f"The directory '{measurement_overlap_dir}' did not exist and has been created.")
                cropped_bbox_info = self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"][f"preprocessing_mode[{self.preprocessing_mode}]"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_bbox_info"]
                cropped_standarized_depth_compressed_measurement_tensor_filename = self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"][f"preprocessing_mode[{self.preprocessing_mode}]"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_standarized_depth_compressed_measurement_tensor_filename"]                
                
                ##############  LOADING cropped_label_tensor   ###############
                cropped_standarized_depth_compressed_measurement_tensor_shape = tuple(self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"][f"preprocessing_mode[{self.preprocessing_mode}]"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_standarized_depth_compressed_measurement_tensor_shape"])
                # Load the raw data into a NumPy array
                loaded_array_cropped_standarized_depth_compressed_measurement = np.fromfile(cropped_standarized_depth_compressed_measurement_tensor_filename, dtype=np.float32)           
                # Reshape the loaded array to the original shape
                loaded_array_cropped_standarized_depth_compressed_measurement = loaded_array_cropped_standarized_depth_compressed_measurement.reshape(cropped_standarized_depth_compressed_measurement_tensor_shape)           
                # Convert back to a PyTorch tensor
                cropped_standarized_depth_compressed_measurement_tensor = torch.from_numpy(loaded_array_cropped_standarized_depth_compressed_measurement)
                ##########################################
                
                #cropped_standarized_depth_compressed_measurement_tensor = torch.load(cropped_standarized_depth_compressed_measurement_tensor_filename, weights_only=True)
                measurement_patches = self.create_patches(cropped_standarized_depth_compressed_measurement_tensor, sample_metadata["ROI"], cropped_bbox_info)

                measurement_patches_dictionary = {}
                # Save patches
                for i, measurement_patch in enumerate(measurement_patches):
                    measurement_tensor_patch, measurement_tensor_patch_coord, roi_adjusted_patch = measurement_patch
                    preprocessed_measurement_patch_filename = os.path.join(f"{sample_id}_depth[{self.model_input_dim[2]}]_patch_size[{self.patch_size}]_overlap[{self.overlap_key}]_coords[{measurement_tensor_patch_coord[0]}x{measurement_tensor_patch_coord[1]}].raw")
                    patch_path = os.path.join(measurement_overlap_dir, preprocessed_measurement_patch_filename)
                    # Convert the tensor to numpy.array so the patch values can be stored in raw binary format with .tofile()
                    measurement_tensor_patch_array = measurement_tensor_patch.numpy()
                    measurement_tensor_patch_array.tofile(patch_path)
                    #torch.save(measurement_tensor_patch_contiguous, patch_path)
                    measurement_patches_dictionary["_".join(map(str, measurement_tensor_patch_coord))] = {
                        "patch_coord": measurement_tensor_patch_coord,
                        "patch_path": patch_path,
                        "patch_tensor_shape": list(measurement_tensor_patch.shape),
                        "roi_adjusted_patch": roi_adjusted_patch
                    }
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "measurement", f"depth[{self.model_input_dim[2]}]", f"preprocessing_mode[{self.preprocessing_mode}]", "patches", f"patch_size[{self.patch_size}]", f"overlap[{self.overlap_key}]", measurement_patches_dictionary)  

            label_patches_dict = self.preprocessed_info_dict[sample_id]["label"]["patches"][f"patch_size[{self.patch_size}]"][f"overlap[{self.overlap_key}]"]
            measurement_patches_dict = self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"][f"preprocessing_mode[{self.preprocessing_mode}]"]["patches"][f"patch_size[{self.patch_size}]"][f"overlap[{self.overlap_key}]"]
            for (key1, value1), (key2, value2) in zip(label_patches_dict.items(), measurement_patches_dict.items()):
                self.preprocessed_patches_dataset.append(
                    {
                        "sample_id": sample_id,
                        "coord": key1, # Key 1 and Key 2 are equal
                        "patch_label_info": value1,
                        "patch_measurement_info": value2
                    }
                )

            end_sample_time = time.time()  # ⏱️ Fin del contador
            elapsed_sample_time = end_sample_time - start_sample_time

            print()
            print(f"{sample_id} loaded (preprocessing took {elapsed_sample_time:.2f} seconds)")
            print()
            print(f"{'=' * 40}")
            print()

        self.update_preprocessed_info() #Updates "preprocessed_info.json"

        
        print()
        print(f"==> DATA PREPOCESSING FINISHED")
        print()
        end_time = datetime.now()
        print("Data preprocessing ended at:", end_time)
        print()
        
        # Compute the total execution time
        execution_time = end_time - start_time
        print(f"Total execution time for data preprocessing: {execution_time}")
        print()


    def load_preprocessed_info(self):
        # If it exists, attempt to load existing data from the file
        with open(self.preprocessed_info_json_path, 'r') as file:
            try:
                preprocessed_info_dict = json.load(file)  # Load JSON data into a dictionary
            except json.JSONDecodeError:
                raise ValueError(
                    f"The JSON file '{self.preprocessed_info_json_path}' is corrupted or not formatted correctly."
                )
        return preprocessed_info_dict


    def update_preprocessed_info(self):
        ## Rewrite JSON in disk
        with open(self.preprocessed_info_json_path, 'w') as file:
            json.dump(self.preprocessed_info_dict, file, indent=4)  # The indent parameter makes it human-readable

        print(f"The JSON file '{self.preprocessed_info_json_path}' has been updated.")


    def update_dictionary(self, dictionary, *args):
        # If it's just a key-value pair
        if len(args) == 2:
            if args[0] not in dictionary:
                dictionary[args[0]] = args[1]
            else:
                dictionary[args[0]].update(args[1])
        else:
            current_level = dictionary
            # Iterate through all but the last argument to handle nested keys
            for key in args[:-2]:
                current_level = current_level[key]  # Move deeper into the nested dictionary

            # The last two arguments are the final key-value pair to update
            final_key, final_value = args[-2], args[-1]

            if final_key not in current_level:
                current_level[final_key] = final_value
            else:
                current_level[final_key].update(final_value)


    def create_patches(self, cropped_tensor, ROI_coordinates, cropped_bbox_info):

        for i in range(2):  # Check for i=0 (height) and i=1 (width)
            cropped_dim = cropped_tensor.shape[i + 1]  # i+1 due to channel in dim 0
            required_dim = self.model_input_dim[i]

            if cropped_dim - required_dim == 0:
                continue  # Dimension matches exactly; no issue
            elif cropped_dim - required_dim > 0:
                continue  # Cropped dimension is larger than required
            else:
                raise ValueError(
                    f"The cropped ROI area is too small for the predefined patch size. "
                    f"Dimension {i} mismatch: cropped ROI area dimension {i} = {cropped_dim} "
                    f"vs. patch dimension {i} = {required_dim}."
                )

        slices_per_dim = {}

        for i in range(2):  # Check for i=0 (height) and i=1 (width)
            stride = int(self.model_input_dim[i] * (1-self.overlap[i]))
            #print(f"Dim {i} - Stride: {stride}")
            #print(f"Dim {i} - Model Input Dim: {self.model_input_dim[i]}")
            #print(f"Dim {i} - Cropped Tensor Shape: {cropped_tensor.shape[i + 1]}")
            slices = []
            for j in range(0, cropped_tensor.shape[i + 1], stride):
                slice_start = j
                slice_end = j + self.model_input_dim[i]
                if slice_end <= cropped_tensor.shape[i + 1]:
                    slice_to_add = slice(slice_start, slice_end)
                    #print(f"Dim {i} - Slice: {slice_to_add}")
                    slices.append(slice_to_add)
                else:
                    out_of_bounds = slice_end - cropped_tensor.shape[i + 1]
                    slice_to_add = slice(slice_start - out_of_bounds, slice_end - out_of_bounds)
                    if not slice_to_add in slices:
                      #print(f"Dim {i} - Slice: {slice_to_add}")
                      slices.append(slice_to_add)
                    else:
                      break
            slices_per_dim[i] = slices

        # Get the lists of slices
        slices_height_dim = slices_per_dim[0]
        slices_width_dim = slices_per_dim[1]

        # Generate all combinations (Cartesian product) of slices between both dimensions
        patches_slices = list(product(slices_height_dim, slices_width_dim))

        # List to store the resulting patches
        patches_list = []

        # Iterate over all the patches
        for patch_slices in patches_slices:
            patch_slice_height_dim = patch_slices[0]
            patch_slice_width_dim = patch_slices[1]

            roi_start = (patch_slice_height_dim.start, patch_slice_width_dim.start)

            if len(cropped_tensor.shape) == 3: # Label tensor
                patch_tensor = cropped_tensor[:, patch_slice_height_dim, patch_slice_width_dim]
            else: # Measurement tensor
                patch_tensor = cropped_tensor[:, patch_slice_height_dim, patch_slice_width_dim, :]

            roi_coords_in_cropped_tensor = {
                'all_points_x': [x - cropped_bbox_info["x_coord"] for x in ROI_coordinates['all_points_x']],
                'all_points_y': [y - cropped_bbox_info["y_coord"] for y in ROI_coordinates['all_points_y']]
            }

            roi_adjusted_patch = {
                'all_points_x': [x - roi_start[1] for x in roi_coords_in_cropped_tensor['all_points_x']],
                'all_points_y': [y - roi_start[0] for y in roi_coords_in_cropped_tensor['all_points_y']]
            }
            # Append the cropped tensor and adjusted ROI to the list
            patches_list.append((patch_tensor, (roi_start[0], roi_start[1]), roi_adjusted_patch))

        if self.include_plotting: 
            if len(cropped_tensor.shape) == 3: # Label tensor
                    # Plotting Label Tensor Patches
                    self.plot_patches(
                        patches_list,
                        tensor_type='Label',
                        channel=1,  # Specify the label channel (e.g., foreground)
                        cmap='RdBu',
                        number_slices_height_dim=len(slices_height_dim),
                        number_slices_width_dim=len(slices_width_dim)
                    )
            else: # Measurement tensor
                    # Plotting Measurement Tensor Patches
                    self.plot_patches(
                        patches_list,
                        tensor_type='Measurement',
                        depth_frame=15,  # Specify the frame index
                        cmap='RdBu',
                        number_slices_height_dim=len(slices_height_dim),
                        number_slices_width_dim=len(slices_width_dim)
                    )


        return patches_list

    def load_and_crop_label(self, measurement_id, measurement_data):
        measurement_label_filename = measurement_data["label_filename"]
        measurement_ROI = measurement_data["ROI"]

        # ############# LABEL LOADING ###################

        # Load label data
        label_img_ndarray = mpimg.imread(os.path.join(os.getcwd(), self.data_dir, "labels", measurement_label_filename))
        
        if len(label_img_ndarray.shape) == 3:
            label_img_ndarray = label_img_ndarray[..., 0] # Shape (256, 320)
        
        # ############# LABEL CONVERSION TO MULTI-CHANNEL (IT IS ADAPTED TO THE MODEL) ###################

        # Convert label to multi-channel for the model
        label_one_hot_encoded, label_mapping = self.one_hot_encode(label_img_ndarray)
        label_tensor = label_one_hot_encoded # Shape (2, 256, 320)

        if self.include_plotting:
            self.plot_tensor_and_polygon(label_tensor, measurement_ROI, "Label Tensor")

        # ############# CROPPING LABEL USING MANUALLY DEFINED ROI ###################

        #print(f"(Before cropping) label_tensor.shape: {label_tensor.shape}")

        # 1. Extract bounding box coordinates from ROI polygon
        x_coords = np.array(measurement_ROI['all_points_x'])
        y_coords = np.array(measurement_ROI['all_points_y'])

        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Calculate width and height ob cropped bbox
        width_bbox = x_max - x_min
        height_bbox = y_max - y_min

        if height_bbox < self.model_input_dim[0]:
            y_c = (y_min + y_max) / 2
            # Calculate new coordinates while keeping the center fixed
            y_min = int(y_c - self.model_input_dim[0] / 2)
            y_max = int(y_c + self.model_input_dim[0] / 2)
    
        if width_bbox < self.model_input_dim[1]:
            x_c = (x_min + x_max) / 2
            # Calculate new coordinates while keeping the center fixed
            x_min = int(x_c - self.model_input_dim[1] / 2)
            x_max = int(x_c + self.model_input_dim[1] / 2)

        if x_min < 0 or x_max > label_tensor.shape[2] or y_min < 0 or y_max > label_tensor.shape[1]:
            raise ValueError(f"Bounding box slice [height, width] ({y_min}:{y_max}, {x_min}:{x_max}) falls outside the image limits [height, width] (0:{label_tensor.shape[1]}, 0:{label_tensor.shape[2]}).")

        # 2. Crop the tensor along height and width using the bounding box
        # Slicing along height (y-axis) and width (x-axis)
        cropped_label_tensor = label_tensor[:, y_min:y_max, x_min:x_max]

        cropped_bbox_info = {
            'y_coord': y_min,
            'x_coord': x_min,
            'height': y_max-y_min,
            'width': x_max-x_min,
        }

        ################### LOGGING #############################

        #print(f"(After cropping) cropped_label_tensor.shape: {cropped_label_tensor.shape}")

        roi_coords_in_cropped_tensor = {
            'all_points_x': [x - x_min for x in measurement_ROI['all_points_x']],
            'all_points_y': [y - y_min for y in measurement_ROI['all_points_y']]
        }

        if self.include_plotting:
            self.plot_tensor_and_polygon(cropped_label_tensor, roi_coords_in_cropped_tensor,"(Cropped) Label Tensor")

        return cropped_label_tensor, cropped_bbox_info

    def load_crop_standardize_depth_compress_measurement(self, measurement_id, measurement_data):
        measurement_3D_thermal_sequence_filename = measurement_data["3D_thermal_sequence_filename"]
        measurement_ROI = measurement_data["ROI"]

        # ############# 3D SEQUENCE DATA LOADING ###################

        # Load 3D sequence data
        mat_data = io.loadmat(os.path.join(os.getcwd(), self.data_dir, "data", measurement_3D_thermal_sequence_filename))
        measurement_3D_thermal_sequence = np.float32(mat_data["imageArray"])  # Shape (256, 320, 1810)
        measurement_tensor = torch.tensor(measurement_3D_thermal_sequence).unsqueeze(0)  # Shape (1, 256, 320, 1810)

        if self.include_plotting:
            self.plot_tensor_and_polygon(measurement_tensor[:,:,:,100], measurement_ROI, "Measurement Tensor (Frame 100)")

        # ############# CROPPING MANUALLY DEFINED ROI ###################

        #print(f"(Before cropping) measurement_tensor.shape: {measurement_tensor.shape}")

        # 1. Extract bounding box coordinates from ROI polygon
        x_coords = np.array(measurement_ROI['all_points_x'])
        y_coords = np.array(measurement_ROI['all_points_y'])

        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Calculate width and height ob cropped bbox
        width_bbox = x_max - x_min
        height_bbox = y_max - y_min

        if height_bbox < self.model_input_dim[0]:
            y_c = (y_min + y_max) / 2
            # Calculate new coordinates while keeping the center fixed
            y_min = int(y_c - self.model_input_dim[0] / 2)
            y_max = int(y_c + self.model_input_dim[0] / 2)
    
        if width_bbox < self.model_input_dim[1]:
            x_c = (x_min + x_max) / 2
            # Calculate new coordinates while keeping the center fixed
            x_min = int(x_c - self.model_input_dim[1] / 2)
            x_max = int(x_c + self.model_input_dim[1] / 2)

        if x_min < 0 or x_max > measurement_tensor.shape[2] or y_min < 0 or y_max > measurement_tensor.shape[1]:
            raise ValueError(f"Bounding box slice [height, width] ({y_min}:{y_max}, {x_min}:{x_max}) falls outside the image limits [height, width] (0:{measurement_tensor.shape[1]}, 0:{measurement_tensor.shape[2]}).")

        # 2. Crop the tensor along height and width using the bounding box
        # Slicing along height (y-axis) and width (x-axis)
        cropped_measurement_tensor = measurement_tensor[:, y_min:y_max, x_min:x_max, :]

        cropped_bbox_info = {
            'y_coord': y_min,
            'x_coord': x_min,
            'height': y_max-y_min,
            'width': x_max-x_min,
        }

        ################### LOGGING #############################

        #print(f"(After cropping) cropped_measurement_tensor.shape: {cropped_measurement_tensor.shape}")

        roi_coords_in_cropped_tensor = {
            'all_points_x': [x - x_min for x in measurement_ROI['all_points_x']],
            'all_points_y': [y - y_min for y in measurement_ROI['all_points_y']]
        }

        if self.include_plotting:
            self.plot_tensor_and_polygon(cropped_measurement_tensor[:,:,:,100], roi_coords_in_cropped_tensor, "(Cropped) Measurement Tensor\n(Frame 100)")

        # ############# NORMALIZE 3D SEQUENCE (3D STANDARDIZATION) ###################

        # Standardize the volume channel-wise
        # Mean and std are calculated along the spatial and depth dimensions (H, W, D)
        mean = cropped_measurement_tensor.mean(dim=(1, 2, 3), keepdim=True)  # Keep dimensions for broadcasting
        std = cropped_measurement_tensor.std(dim=(1, 2, 3), keepdim=True)

        # Standardize: (value - mean) / std
        cropped_normalized_measurement_tensor = (cropped_measurement_tensor - mean) / std

        #print("(Before normalization/standardization) cropped_measurement_tensor.shape:", cropped_measurement_tensor.shape)
        #print("(After normalization/standardization) cropped_normalized_measurement_tensor.shape:", cropped_normalized_measurement_tensor.shape)

        if self.include_plotting:
            self.plot_tensor_and_polygon(cropped_normalized_measurement_tensor[:,:,:,100], roi_coords_in_cropped_tensor, "(Cropped) 3D_Standardized Measurement Tensor\n(Frame 100)")

        # ############# TEMPORAL COMPRESSION ###################

        model_input_depth_dim = self.model_input_dim[2]

        cropped_normalized_compressed_measurement_tensor = self.compress_depth_tensor(cropped_normalized_measurement_tensor, model_input_depth_dim)

        #print(f"(Before compression) cropped_normalized_measurement_tensor.shape: {cropped_normalized_measurement_tensor.shape}")
        #print(f"(After compression) cropped_normalized_compressed_measurement_tensor.shape: {cropped_normalized_compressed_measurement_tensor.shape}")

        if self.include_plotting:
            self.plot_tensor_and_polygon(cropped_normalized_compressed_measurement_tensor[:,:,:,15], roi_coords_in_cropped_tensor, f"(Cropped) 3D_Standardized/Compressed_{model_input_depth_dim} Measurement Tensor\n(Frame 15 After Compression)")

        return cropped_normalized_compressed_measurement_tensor, cropped_bbox_info

    def plot_patches(self, patches, tensor_type, number_slices_height_dim, number_slices_width_dim, channel=None, depth_frame=None, cmap='RdBu'):
        """
        Plots the patches of measurement or label tensors with adjusted ROI polygons.

        Args:
            patches (list): List of tuples [(patch_tensor, patch_coord, adjusted_roi)].
            tensor_type (str): Type of tensor ('Measurement' or 'Label').
            channel (int, optional): Channel index to visualize (for label tensors).
            depth_frame (int, optional): Frame index to visualize (for measurement tensors).
            cmap (str): Colormap for visualization.
        """
        # Maximum number of patches per row
        max_patches_per_row = 5
        
        # Calculate the number of patches
        total_patches = len(patches)
        
        # Determine the number of rows and patches per row
        patches_per_row = min(total_patches, max_patches_per_row)
        n_rows = (total_patches + max_patches_per_row - 1) // max_patches_per_row  # Ceiling division
        
        # Create the subplots
        fig, axes = plt.subplots(n_rows, patches_per_row, figsize=(patches_per_row * 5, n_rows * 5))
        
        # Ensure axes is always a 1D array, even if there's only one row or one patch
        if n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (patch_tensor, patch_coord, adjusted_roi) in enumerate(patches):
            ax = axes[idx]
        
            # Determine what to plot based on tensor type
            if tensor_type == 'Measurement' and depth_frame is not None:
                data = patch_tensor[0, :, :, depth_frame].cpu().numpy()  # Frame-specific
                im = ax.imshow(data, cmap=cmap)  # Plot the patch tensor
            elif tensor_type == 'Label' and channel is not None:
                data = patch_tensor[channel, :, :].cpu().numpy()  # Channel-specific
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)  # Plot the patch tensor
            else:
                raise ValueError("Specify 'depth_frame' for Measurement or 'channel' for Label tensors.")
        
            # Set the title
            ax.set_title(f'{tensor_type} Patch Coord: ({patch_coord[0]}, {patch_coord[1]})')
            ax.axis('off')
            fig.colorbar(im, ax=ax, label='Pixel Value')
        
            # Plot the adjusted polygon overlay
            all_points_x = adjusted_roi['all_points_x']
            all_points_y = adjusted_roi['all_points_y']
            ax.plot(all_points_x + [all_points_x[0]], all_points_y + [all_points_y[0]], 'r-', linewidth=2)  # Close the polygon
            ax.scatter(all_points_x, all_points_y, color='blue', zorder=5)  # Mark the vertices
        
        # Hide any unused axes
        for idx in range(total_patches, len(axes)):
            axes[idx].axis('off')
        
        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show() 


    def plot_tensor_and_polygon(self,tensor, roi, name_plot_tensor):
        """
        Plot the mask and ROI polygon together for visualization.

        Parameters:
        - mask (torch.Tensor): The mask to be plotted.
        - roi (dict): Dictionary with keys 'all_points_x' and 'all_points_y' representing the ROI polygon.
        """
        if tensor.shape[0] == 1:
            # Plotting the mask
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(tensor[0].cpu().numpy(), cmap='RdBu')
            plt.colorbar(im, ax=ax, label='Pixel Value')

            # Correct the y-coordinates to match the image grid
            all_points_x = roi['all_points_x']
            all_points_y = roi['all_points_y']
            plt.plot(all_points_x + [all_points_x[0]], all_points_y + [all_points_y[0]], 'r-', linewidth=2)  # Close the polygon
            plt.scatter(all_points_x, all_points_y, color='blue', zorder=5)  # Mark the vertices
            plt.suptitle(f'{name_plot_tensor} with ROI Polygon Overlay')
            plt.show()
        else:
            fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, 6))
            for i in range(tensor.shape[0]):
                im = axes[i].imshow(tensor[i].cpu().numpy(), cmap='RdBu')
                axes[i].set_title(f'Channel {i}')
                fig.colorbar(im, ax=axes[i], label='Pixel Value')

                # Add polygon overlay
                all_points_x = roi['all_points_x']
                all_points_y = roi['all_points_y']
                axes[i].plot(all_points_x + [all_points_x[0]], all_points_y + [all_points_y[0]], 'r-', linewidth=2)  # Close the polygon
                axes[i].scatter(all_points_x, all_points_y, color='blue', zorder=5)

            plt.suptitle(f'{name_plot_tensor} with ROI Polygon Overlay')
            plt.show()

    def compress_depth_tensor(self, tensor, model_input_dim_depth):
        num_channels, height, width, depth = tensor.shape
        
        #"normal_interval_subsampling","high_temp_frame_interval_subsampling"
        if self.preprocessing_mode == "hotspot_centered_gaussian_smoothed_equidistant_subsampling":
            # Ensure num_channels is 1
            assert num_channels == 1, "num_channels must be 1"
            
            # Obtain the avg temperature for each frame
            avg_temp_per_frame = torch.mean(tensor, dim=(0, 1, 2))  
            
            # Find the time/frame index with the highest temperature
            highest_temp_frame_index = torch.argmax(avg_temp_per_frame).item()
            
            # Define the interval length
            interval_length = 700
            half_interval = interval_length // 2
            
            # Calculate the initial start and end
            start = highest_temp_frame_index - half_interval
            end = start + interval_length
            
            # Adjust the interval if it falls out of bounds
            if start < 0:
                start = 0
                end = interval_length
            
            if end > depth:
                end = time
                start = time - interval_length

            if end > depth or start < 0:
                raise ValueError(
                                f"The input tensor depth [{depth}] is smaller than the temporal dimension to which the data wants to be compressed [{interval_length}]"
                                f" in preprocessing_mode: [high_temp_frame_interval_subsampling]"
                            )
            
            # Slice the tensor for the time interval
            tensor = tensor[:, :, :, start:end]

        elif self.preprocessing_mode == "uniform_gaussian_smoothed_equidistant_subsampling":
            tensor = tensor
        else:
            raise ValueError(
                                f"Invalid configuration value: {self.preprocessing_mode}. "
                                f"Allowed values are: ['normal_interval_subsampling', 'high_temp_frame_interval_subsampling']"
                            )

        num_channels, height, width, depth = tensor.shape
            
        # Raise an exception if depth is smaller than model_input_dim_depth
        if depth < model_input_dim_depth:
            raise ValueError(f"The depth of the input tensor ({depth}) must be greater than or equal to model_input_dim_depth ({model_input_dim_depth}).")

        block_size = depth / model_input_dim_depth

        # Convert to NumPy
        numpy_array = tensor.numpy()

        # Apply Gaussian filter only along the last dimension (depth)
        # sigma = block_size / 2 beacuse is the number of neighbours we look right and left
        neighbor_one_side = block_size / 2
        smoothed_numpy_array = gaussian_filter(numpy_array, sigma=(0, 0, 0, neighbor_one_side / 3))

        # Convert back to PyTorch tensor
        smoothed_tensor = torch.tensor(smoothed_numpy_array, dtype=torch.float32)

        # Generate fractional indices
        indices = torch.linspace(0, depth - 1, model_input_dim_depth)
        # print(f"Indices shape: {indices.shape}")
        # print(f"Indices: {indices}")

        # Round to nearest integer and clamp indices to valid range
        indices = torch.clamp(indices.round().long(), 0, depth - 1)

        # Select slices at these indices
        reduced_tensor = smoothed_tensor[..., indices]

        return reduced_tensor

    def one_hot_encode(self, array):
        # Get unique labels in the array
        unique_labels = np.unique(array)
        # Create a dictionary mapping each label to an index
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        # Shape for one-hot encoding: (height, width, num_classes)
        one_hot_shape = array.shape + (len(unique_labels),)
        # Initialize one-hot encoded array
        one_hot_encoded = np.zeros(one_hot_shape, dtype=np.float32)
        # Populate one-hot array
        for label, index in label_to_index.items():
            one_hot_encoded[..., index] = (array == label).astype(np.float32)
        #Convert into tensor
        one_hot_encoded = torch.tensor(one_hot_encoded, dtype=torch.float32)
        # Shape for one-hot encoding: (num_classes, height, width)
        one_hot_encoded = one_hot_encoded.permute(2, 0, 1)

        return one_hot_encoded, label_to_index

    def __len__(self):
        if self.is_inference_mode:
            return len(self.metadata_dict_with_files_selected)
        else:
            return len(self.preprocessed_patches_dataset)

    def __getitem__(self, idx):
        if self.is_inference_mode:
            # sample_id = list(self.metadata_dict_with_files_selected.keys())[idx]
            # y = y_filename = self.preprocessed_info_dict[sample_id]["label"]["cropped_label_tensor_filename"]
            # x = patches_info_dict = self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"]["patches"][f"patch_size[{self.patch_size}]"][f"overlap[{self.overlap_key}]"]
            
            sample_id = list(self.metadata_dict_with_files_selected.keys())[idx]
            y_filename = self.preprocessed_info_dict[sample_id]["label"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_label_tensor_filename"]
            y_tensor_shape = tuple(self.preprocessed_info_dict[sample_id]["label"]["patches"][f"patch_size[{self.patch_size}]"]["cropped_label_tensor_shape"])
            y = (y_filename, y_tensor_shape)
            x = patches_info_dict = self.preprocessed_info_dict[sample_id]["measurement"][f"depth[{self.model_input_dim[2]}]"][f"preprocessing_mode[{self.preprocessing_mode}]"]["patches"][f"patch_size[{self.patch_size}]"][f"overlap[{self.overlap_key}]"]
            return sample_id, x, y
        else:
            # x_filename = self.preprocessed_patches_dataset[idx]["patch_measurement_info"]["patch_path"]
            # y_filename = self.preprocessed_patches_dataset[idx]["patch_label_info"]["patch_path"]

            # x = torch.load(x_filename, weights_only=True)
            # y = torch.load(y_filename, weights_only=True)

            x_filename = self.preprocessed_patches_dataset[idx]["patch_measurement_info"]["patch_path"]
            y_filename = self.preprocessed_patches_dataset[idx]["patch_label_info"]["patch_path"]

            x_tensor_shape = tuple(self.preprocessed_patches_dataset[idx]["patch_measurement_info"]["patch_tensor_shape"])
            y_tensor_shape = tuple(self.preprocessed_patches_dataset[idx]["patch_label_info"]["patch_tensor_shape"])

            # Load the raw data into a NumPy array
            loaded_array_x = np.fromfile(x_filename, dtype=np.float32)
            loaded_array_y = np.fromfile(y_filename, dtype=np.float32)
            
            # Reshape the loaded array to the original shape
            loaded_array_x = loaded_array_x.reshape(x_tensor_shape)
            loaded_array_y = loaded_array_y.reshape(y_tensor_shape)
            
            # Convert back to a PyTorch tensor
            x = torch.from_numpy(loaded_array_x)
            y = torch.from_numpy(loaded_array_y)

            return x, y