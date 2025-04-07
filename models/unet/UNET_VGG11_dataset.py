import os                           # For file system operations like path manipulation and directory creation
import json                         # For loading and saving JSON files
import numpy as np                  # For numerical operations and array manipulations
import torch                        # For handling tensors
from torch.utils.data import Dataset  # For creating custom PyTorch datasets
import matplotlib.pyplot as plt     # For plotting images and visualizations
import matplotlib.image as mpimg    # For reading images with matplotlib
from itertools import product       # For generating Cartesian products (used if combinatorial operations are added)
from scipy.ndimage import gaussian_filter  # For applying Gaussian filters (used if preprocessing filters are applied)
import scipy.io as io               # For loading MATLAB files (.mat format)
import math                         # For mathematical operations like rounding up to multiples of 32
from multiprocessing import Pool    # For parallel processing of time series data
import time


print("All imports succeeded!")


class UNET_VGG11_Dataset(Dataset):
    def __init__(self, augmentation: bool, metadata_dict_with_files_selected: dict, data_dir: str, preprocessing_technique: str = "pct", preprocessing_channels: int = 10 , include_plotting: bool = False):
        """
        Initializes the class instance with the provided parameters.

        Parameters:
        metadata_dict_with_files_selected (dict): Dictionary containing the selected files for the dataset with their metadata (e.g., 3D_thermal_sequence_filename, label_filename, ROI, stratified group, etc.).
        data_dir (str): Directory where the data is located.
        model_input_dim (tuple): 3D tuple indicating the dimensions of the model input (height, width, depth).
        """

        self.metadata_dict_with_files_selected = metadata_dict_with_files_selected
        self.data_dir = data_dir

        self.augmentation = augmentation

        self.include_plotting = include_plotting

        self.preprocessing_technique = preprocessing_technique
        if not self.preprocessing_technique in ["pct","ppt"]:
            raise ValueError("Wrong preprocessing_technique argument: value not valid. 'pct' or 'ppt' only valid values.")
        self.preprocessing_channels = preprocessing_channels

        # self.model_input_dim = model_input_dim #e.g. (64,64,128)

        # self.overlap = overlap #e.g. [0.25,0.35]
        # formatted_overlap_dim_0 = f"{self.overlap[0]:.2f}"
        # formatted_overlap_dim_1 = f"{self.overlap[1]:.2f}"

        # self.patch_size = f"{self.model_input_dim[0]}x{self.model_input_dim[1]}" #e.g. 64x64
        # self.overlap_key = f"{formatted_overlap_dim_0.replace('.', '_')}x{formatted_overlap_dim_1.replace('.', '_')}" #e.g. 0_25x0_35

        self.preprocessed_dir = os.path.join(os.getcwd(), self.data_dir, "preprocessed_files")

        self.preprocessed_samples_dataset = []

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

        print(f"==> DATA PREPOCESSING AND LOADING STARTS")
        print()

        if not os.path.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            print(f"The directory '{self.preprocessed_dir}' did not exist and has been created.")
            print()

        self.preprocessed_dir = os.path.join(self.preprocessed_dir, "unet")
        if not os.path.isdir(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            print(f"The directory '{self.preprocessed_dir}' did not exist and has been created.")
            print()

        self.preprocessed_info_json_path = os.path.join(self.preprocessed_dir, "preprocessed_info.json")
        self.preprocessed_info_dict = None
        if os.path.isfile(self.preprocessed_info_json_path):
            self.preprocessed_info_dict = self.load_preprocessed_info()
        else:
            self.preprocessed_info_dict = {}

        for sample_id, sample_metadata in self.metadata_dict_with_files_selected.items():

            print(f"Processing & loading sample {sample_id} ...")
            print(f"{'=' * 40}")
            print()

            start_time = time.time()  # ⏱️ Iniciar contador
            
            # Checks
            if not sample_id in self.preprocessed_info_dict:
                # Create directory for the sample
                preprocessed_sample_dir = os.path.join(self.preprocessed_dir, f"{sample_id}")
                os.makedirs(preprocessed_sample_dir)
                print(f"The directory '{preprocessed_sample_dir}' did not exist and has been created.")
    
                # Create and save the cropped label in the directory just created
                cropped_label_tensor, cropped_bbox_info = self.load_and_crop_label(sample_id, sample_metadata)
                cropped_label_tensor_filename = os.path.join(preprocessed_sample_dir,f"{sample_id}_cropped_label.raw")
                
                # Convert the tensor to numpy.array so the patch values can be stored in raw binary format with .tofile()
                cropped_label_tensor_array = cropped_label_tensor.numpy()
                cropped_label_tensor_array.tofile(cropped_label_tensor_filename)
                print(f"File was generated and saved successfully: {cropped_label_tensor_filename}")
                
                #torch.save(cropped_label_tensor, cropped_label_tensor_filename)
                # Update the information in the preprocessed_info_json
                self.update_dictionary(self.preprocessed_info_dict, sample_id, {})
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "cropped_bbox_info", cropped_bbox_info)
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "cropped_label_tensor_filename", cropped_label_tensor_filename)
                self.update_dictionary(self.preprocessed_info_dict, sample_id, "cropped_label_tensor_shape", list(cropped_label_tensor.shape))

                # Create and save the cropped, standarized & depth compressed 3D measurement in the directory just created
                cropped_pct_preprocessed_measurement_tensor, cropped_ppt_preprocessed_measurement_tensor  = self.load_crop_preprocess_measurement(sample_id, sample_metadata, self.preprocessing_channels)
                cropped_pct_preprocessed_measurement_tensor_filename = os.path.join(preprocessed_sample_dir, f"{sample_id}_cropped_preprocessed[pct]_channels[{self.preprocessing_channels}].raw")
                cropped_ppt_preprocessed_measurement_tensor_filename = os.path.join(preprocessed_sample_dir, f"{sample_id}_cropped_preprocessed[ppt]_channels[{self.preprocessing_channels}].raw")
                # Convert the tensor to numpy.array so the patch values can be stored in raw binary format with .tofile()
                cropped_pct_preprocessed_measurement_tensor_array = cropped_pct_preprocessed_measurement_tensor.numpy()
                cropped_pct_preprocessed_measurement_tensor_array.tofile(cropped_pct_preprocessed_measurement_tensor_filename)
                print(f"File was generated and saved successfully: {cropped_pct_preprocessed_measurement_tensor_filename}")
                cropped_ppt_preprocessed_measurement_tensor_array = cropped_ppt_preprocessed_measurement_tensor.numpy()
                cropped_ppt_preprocessed_measurement_tensor_array.tofile(cropped_ppt_preprocessed_measurement_tensor_filename)
                print(f"File was generated and saved successfully: {cropped_ppt_preprocessed_measurement_tensor_filename} \n")
                #torch.save(cropped_standarized_depth_compressed_measurement_tensor, cropped_standarized_depth_compressed_measurement_tensor_filename)
                # Update the information in the preprocessed_info_json
                self.update_dictionary(self.preprocessed_info_dict, sample_id, f"cropped_preprocessed[pct]_measurement_tensor_filename", cropped_pct_preprocessed_measurement_tensor_filename)
                self.update_dictionary(self.preprocessed_info_dict, sample_id, f"cropped_preprocessed[ppt]_measurement_tensor_filename", cropped_ppt_preprocessed_measurement_tensor_filename)
                self.update_dictionary(self.preprocessed_info_dict, sample_id, f"cropped_preprocessed_measurement_tensor_shape", list(cropped_pct_preprocessed_measurement_tensor.shape))


            end_time = time.time()  # ⏱️ Fin del contador
            elapsed_time = end_time - start_time

            print()
            print(f"{sample_id} preprocessed (preprocessing took {elapsed_time:.2f} seconds)")
            print()
            print(f"{'=' * 40}")
            print()

        self.update_preprocessed_info() #Updates "preprocessed_info.json"

        print()
        print(f"==> DATA PREPOCESSING AND LOADING FINISHED")
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

    # def adjust_bbox(self, x_min, y_min, x_max, y_max):
    #     """
    #     Expands a bounding box so its width and height are divisible by 32 while keeping it centered.
    
    #     Parameters:
    #         x_min (float): Original top-left x-coordinate.
    #         y_min (float): Original top-left y-coordinate.
    #         x_max (float): Original bottom-right x-coordinate.
    #         y_max (float): Original bottom-right y-coordinate.
    
    #     Returns:
    #         (float, float, float, float): New expanded bounding box coordinates (x_min', y_min', x_max', y_max').
    #     """
    #     # Calculate original center
    #     x_c = (x_min + x_max) / 2
    #     y_c = (y_min + y_max) / 2
    
    #     # Calculate original width and height
    #     w = x_max - x_min
    #     h = y_max - y_min
    
    #     # Expand width and height to the nearest multiple of 32
    #     w_new = math.ceil(w / 32) * 32
    #     h_new = math.ceil(h / 32) * 32
    
    #     # Calculate new coordinates while keeping the center fixed
    #     x_min_new = int(x_c - w_new / 2)
    #     x_max_new = int(x_c + w_new / 2)
    #     y_min_new = int(y_c - h_new / 2)
    #     y_max_new = int(y_c + h_new / 2)
    
    #     return x_min_new, y_min_new, x_max_new, y_max_new

    def adjust_bbox(self, x_min, y_min, x_max, y_max):
        """
        Creates a fixed 224x224 bounding box, centered on the original box, 
        with all integer coordinates.
    
        Parameters:
            x_min (float): Original top-left x-coordinate.
            y_min (float): Original top-left y-coordinate.
            x_max (float): Original bottom-right x-coordinate.
            y_max (float): Original bottom-right y-coordinate.
    
        Returns:
            (int, int, int, int): New bounding box (x_min', y_min', x_max', y_max').
        """
        import math
    
        # Compute the original center in floating-point
        x_c = (x_min + x_max) / 2.0
        y_c = (y_min + y_max) / 2.0
    
        # We fix the dimensions to 224 x 224
        half_w = 224 // 2  # = 112
        half_h = 224 // 2  # = 112
    
        # Use round (or floor/ceil) to get the nearest integer center
        x_min_new = int(round(x_c - half_w))
        y_min_new = int(round(y_c - half_h))
    
        # Then the max coordinates follow from the fixed size
        x_max_new = x_min_new + 224
        y_max_new = y_min_new + 224
    
        return x_min_new, y_min_new, x_max_new, y_max_new

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

    def load_and_crop_label(self, sample_id, measurement_data):
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

        cropped_x_min, cropped_y_min, cropped_x_max, cropped_y_max = self.adjust_bbox(x_min, y_min, x_max, y_max)
        
        # 2. Crop the tensor along height and width using the bounding box
        # Slicing along height (y-axis) and width (x-axis)
        cropped_label_tensor = label_tensor[:, cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max]

        cropped_bbox_info = {
            'y_coord': cropped_y_min,
            'x_coord': cropped_x_min,
            'height': cropped_y_max-cropped_y_min,
            'width': cropped_x_max-cropped_x_min,
        }

        ################### LOGGING #############################

        #print(f"(After cropping) cropped_label_tensor.shape: {cropped_label_tensor.shape}")

        roi_coords_in_cropped_tensor = {
            'all_points_x': [x - cropped_x_min for x in measurement_ROI['all_points_x']],
            'all_points_y': [y - cropped_y_min for y in measurement_ROI['all_points_y']]
        }

        if self.include_plotting:
            self.plot_tensor_and_polygon(cropped_label_tensor, roi_coords_in_cropped_tensor,"(Cropped) Label Tensor")

        return cropped_label_tensor, cropped_bbox_info

    ################ PCT PREPROCESSING METHOD ################

    def pct_laval(self, data, preprocessing_channels):
        """
        Principal Component Thermography (PCT)
    
        Parameters:
            data (numpy.ndarray): 3D array of shape (Ny, Nx, Nt)
    
        Returns:
            data_pct (numpy.ndarray): 3D array with principal components of shape (Ny, Nx, Nt)
        """
        # Get the dimensions of the input data
        Ny, Nx, Nt = data.shape
        M = Ny * Nx
    
        # Reshape the data to a 2D array (M, Nt)
        pixel_time = data.reshape(M, Nt)
    
        # Normalize the data (subtract mean and divide by standard deviation)
        mew = np.mean(pixel_time, axis=0)
        sigma = np.std(pixel_time, axis=0)
    
        for m in range(Nt):
            pixel_time[:, m] = (pixel_time[:, m] - mew[m]) / sigma[m]
    
        # Perform Singular Value Decomposition (SVD)
        U, _, _ = np.linalg.svd(pixel_time, full_matrices=False)
    
        # Reshape the result back to a 3D array (Ny, Nx, Nt)
        data_pct = U.reshape(Ny, Nx, Nt)
    
        return data_pct[:,:,:preprocessing_channels]

    ##########################################################
    
    ################ PPT PREPROCESSING METHOD & AUXILIARY FUNCTIONS ################

    def apply_function_to_time_series(self, args):
        """
        Applies a specified function to a single time series.
    
        Parameters:
            args (tuple): A tuple containing:
                - time_series (numpy.ndarray): The time series data.
                - func (callable): The function to apply to the time series.
                - func_args (tuple): Additional positional arguments for the function.
                - func_kwargs (dict): Additional keyword arguments for the function.
    
        Returns:
            numpy.ndarray: The processed time series.
        """
        time_series, func, func_args, func_kwargs = args
        return func(time_series, *func_args, **func_kwargs)
    
    
    def process_time_series_3d(self, data, func, *args, **kwargs):
        """
        Applies a given function to each pixel's time series in a 3D data array using parallel processing.
    
        Parameters:
            data (numpy.ndarray): 3D array of shape (Ny, Nx, Nt), where:
                - Ny: Number of rows (height).
                - Nx: Number of columns (width).
                - Nt: Number of time points.
            func (callable): Function to apply to each time series.
            *args: Additional positional arguments for the function.
            **kwargs: Additional keyword arguments for the function.
    
        Returns:
            numpy.ndarray: Processed 3D array with shape (Ny, Nx, new_time_length),
                           where new_time_length depends on the function's output.
        """
        # Get dimensions
        Ny, Nx, Nt = data.shape
        total_pixels = Ny * Nx
    
        # Reshape the 3D data to a 2D array (total_pixels, Nt)
        reshaped_data = data.reshape(total_pixels, Nt)
    
        # Prepare arguments for each call to apply_function_to_time_series
        task_args = [(reshaped_data[i], func, args, kwargs) for i in range(total_pixels)]
    
        # Apply the function to each time series in parallel
        with Pool() as pool:
            processed_data = pool.map(self.apply_function_to_time_series, task_args)
    
        # Convert the list of results to a NumPy array
        processed_data = np.array(processed_data, dtype=np.float32)
    
        # Reshape back to 3D (Ny, Nx, new_time_length)
        new_time_length = processed_data.shape[1]
        result = processed_data.reshape(Ny, Nx, new_time_length)
    
        return result
    
    
    def compute_fft_phase(self, time_series):
        """
        Computes the phase of the FFT for a single time series.
    
        Parameters:
            time_series (numpy.ndarray): 1D array representing the time series.
    
        Returns:
            numpy.ndarray: 1D array representing the phase of the FFT.
        """
        return np.angle(np.fft.fft(time_series))
    
    
    def compute_fft_phase_3d(self, data, preprocessing_channels):
        """
        Computes the phase of the FFT for each time series in a 3D data array.
    
        Parameters:
            data (numpy.ndarray): 3D array where the last dimension represents time (Ny, Nx, Nt).
    
        Returns:
            numpy.ndarray: 3D array with the phase of the FFT for each time series.
        """
        return self.process_time_series_3d(data, self.compute_fft_phase)[:,:,:preprocessing_channels]

    ########################### NORMALIZATION ##################################

    def normalize_image_percentiles(self, image):
        """
        Normaliza una imagen de múltiples canales entre el percentil 1 y el percentil 99 
        de cada canal independientemente.
        
        Parámetros:
        - image: np.array de forma (H, W, C) donde C es el número de canales.
    
        Retorna:
        - image_normalized: Imagen normalizada de la misma forma que la original.
        """
        image_normalized = np.zeros_like(image, dtype=np.float32)
        
        for channel in range(image.shape[2]):
            # Extraer el canal
            img_channel = image[:, :, channel]
            
            # Calcular el percentil 1 y 99 para el canal
            p1 = np.percentile(img_channel, 1)
            p99 = np.percentile(img_channel, 99)
            
            # Normalizar el canal entre el percentil 1 y el percentil 99
            img_channel_norm = (img_channel - p1) / (p99 - p1)
            
            # Clipping para asegurar que los valores estén en el rango [0, 1]
            img_channel_norm = np.clip(img_channel_norm, 0, 1)
            
            # Guardar el canal normalizado
            image_normalized[:, :, channel] = img_channel_norm
        
        return image_normalized

    ###############################################################################    

    def load_crop_preprocess_measurement(self, sample_id, measurement_data, preprocessing_channels):
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

        cropped_x_min, cropped_y_min, cropped_x_max, cropped_y_max = self.adjust_bbox(x_min, y_min, x_max, y_max)
        
        # 2. Crop the tensor along height and width using the bounding box
        # Slicing along height (y-axis) and width (x-axis)
        cropped_measurement_tensor = measurement_tensor[:, cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max, :]

        ################### LOGGING #############################

        #print(f"(After cropping) cropped_measurement_tensor.shape: {cropped_measurement_tensor.shape}")

        roi_coords_in_cropped_tensor = {
            'all_points_x': [x - cropped_x_min for x in measurement_ROI['all_points_x']],
            'all_points_y': [y - cropped_y_min for y in measurement_ROI['all_points_y']]
        }

        if self.include_plotting:
            self.plot_tensor_and_polygon(cropped_measurement_tensor[:,:,:,100], roi_coords_in_cropped_tensor, "(Cropped) Measurement Tensor\n(Frame 100)")

        # ############# PCT PROCESSING ###################

        cropped_measurement_array = cropped_measurement_tensor.squeeze(0).numpy()
        pct_preprocessed_measurement_array = self.pct_laval(cropped_measurement_array, preprocessing_channels)
        pct_preprocessed_measurement_array = self.normalize_image_percentiles(pct_preprocessed_measurement_array)
        pct_preprocessed_measurement_tensor = torch.tensor(pct_preprocessed_measurement_array)
        pct_preprocessed_measurement_tensor = pct_preprocessed_measurement_tensor.permute(2, 0, 1)

        if self.include_plotting :
            self.plot_channels_and_roi(pct_preprocessed_measurement_array, roi_coords_in_cropped_tensor, f"({sample_id}) PCT: First {preprocessing_channels} Channels")

        # ############# PPT PROCESSING###################

        cropped_measurement_array = cropped_measurement_tensor.squeeze(0).numpy()
        ppt_preprocessed_measurement_array = self.compute_fft_phase_3d(cropped_measurement_array, preprocessing_channels)
        ppt_preprocessed_measurement_array = self.normalize_image_percentiles(ppt_preprocessed_measurement_array)
        ppt_preprocessed_measurement_tensor = torch.tensor(ppt_preprocessed_measurement_array)
        ppt_preprocessed_measurement_tensor = ppt_preprocessed_measurement_tensor.permute(2, 0, 1)

        if self.include_plotting :
            self.plot_channels_and_roi(ppt_preprocessed_measurement_array, roi_coords_in_cropped_tensor, f"({sample_id}) PPT: First {preprocessing_channels} Channels")

        return pct_preprocessed_measurement_tensor, ppt_preprocessed_measurement_tensor


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

    def plot_channels_and_roi(self, data, roi, title, cmap='gray', max_channels_per_row=5):
        """
        Plots the channels of a 3D data array (height, width, channels).
    
        Parameters:
            data (numpy.ndarray): 3D array of shape (height, width, channels).
            title (str): Title of the plot.
            cmap (str): Colormap for displaying the images. Default is 'gray'.
            max_channels_per_row (int): Maximum number of channels to display per row.
        """
        all_points_x = roi['all_points_x']
        all_points_y = roi['all_points_y']
        
        # Get the number of channels
        num_channels = data.shape[2]
        
        # Calculate the number of rows needed (maximum 5 channels per row)
        num_rows = (num_channels + max_channels_per_row - 1) // max_channels_per_row
        num_cols = min(max_channels_per_row, num_channels)
    
        # Create the figure with the appropriate number of rows and columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        fig.suptitle(title, fontsize=16)
    
        # Flatten the axes array for easy iteration (handles both 1D and 2D axes arrays)
        axes = np.atleast_2d(axes).ravel()
    
        # Plot each channel
        for i in range(num_channels):
            axes[i].imshow(data[:, :, i], cmap=cmap)
            axes[i].set_title(f'Channel {i + 1}')
            axes[i].axis('off')  # Hide axes for better visualization

            axes[i].plot(all_points_x + [all_points_x[0]], all_points_y + [all_points_y[0]], 'r-', linewidth=2)  # Close the polygon
            axes[i].scatter(all_points_x, all_points_y, color='blue', zorder=5)
    
        # Hide any unused subplots
        for j in range(num_channels, len(axes)):
            axes[j].axis('off')
    
        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.metadata_dict_with_files_selected)

    def __getitem__(self, idx):
        sample_id = list(self.metadata_dict_with_files_selected.keys())[idx]
        
        y_filename = self.preprocessed_info_dict[sample_id]["cropped_label_tensor_filename"]
        if self.preprocessing_technique == "pct":
            x_filename = self.preprocessed_info_dict[sample_id]["cropped_preprocessed[pct]_measurement_tensor_filename"]
        elif self.preprocessing_technique == "ppt":
            x_filename = self.preprocessed_info_dict[sample_id]["cropped_preprocessed[ppt]_measurement_tensor_filename"]
        
        y_tensor_shape = tuple(self.preprocessed_info_dict[sample_id]["cropped_label_tensor_shape"])
        x_tensor_shape= tuple(self.preprocessed_info_dict[sample_id]["cropped_preprocessed_measurement_tensor_shape"])

        # Load the raw data into a NumPy array
        loaded_array_x = np.fromfile(x_filename, dtype=np.float32)
        loaded_array_y = np.fromfile(y_filename, dtype=np.float32)
        
        # Reshape the loaded array to the original shape
        loaded_array_x = loaded_array_x.reshape(x_tensor_shape)
        loaded_array_y = loaded_array_y.reshape(y_tensor_shape)
        
        # Convert back to a PyTorch tensor
        x = torch.from_numpy(loaded_array_x)
        y = torch.from_numpy(loaded_array_y)

        return sample_id, x, y