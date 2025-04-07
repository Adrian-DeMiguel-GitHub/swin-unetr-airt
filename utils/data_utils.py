from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
from collections import defaultdict
import random

# Load metadata JSON file
def load_metadata(json_file):
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    return metadata

def stratified_split(data_dictionary, test_size=0.2, random_state=42):
    """
    Perform a stratified split on a dataset where groups are defined in a dictionary.

    Parameters:
    - data_dictionary (dict): Dictionary where keys are sample IDs, and values are dictionaries
                              containing 'feature' and 'group' for stratification.
    - test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int): Seed for reproducibility (default is 42).

    Returns:
    - train_files (dict): Dictionary containing the training split.
    - test_files (dict): Dictionary containing the testing split.
    """

    # Extract IDs and stratified group labels from the dictionary
    ids = np.array(list(data_dictionary.keys()))
    stratify_groups = np.array([data_dictionary[id_]['stratified_group'] for id_ in ids])

    # Perform the stratified split
    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, stratify=stratify_groups, random_state=random_state
    )

    # Create dictionaries for train and test sets
    train_files = {id_: data_dictionary[id_] for id_ in train_ids}
    test_files = {id_: data_dictionary[id_] for id_ in test_ids}

    print()
    print(f"{'=' * 74}")
    print(f"{'=' * 20} {(1-test_size)*100}-{test_size*100} TRAINING/TESTING SPLIT {'=' * 20}")
    print(f"{'=' * 74}")
    print()
    print()

    print(f"{'=' * 10}> TRAINING/VALIDATION SET")
    print()

    print(f" SAMPLES: {train_files.keys()}")
    print()

    print(f"{'=' * 10}> TESTING SET")
    print()

    print(f" SAMPLES: {test_files.keys()}")

    return train_files, test_files

# Stratified K-Fold Split based on a dictionary where the data to be splitted is described in the same format as the metadata json
def k_fold_stratified_split(data_dictionary, k, n_samples_per_group=None, random_state=42):
    # Extract IDs and stratification groups from the data dictionary
    ids = list(data_dictionary.keys())  # R_002, R_003, etc.
    stratify_groups = [data_dictionary[id_]["stratified_group"] for id_ in ids]  # Use 'stratified_group'

    # Group the IDs by their stratified groups
    grouped_ids = defaultdict(list)
    for id_, group in zip(ids, stratify_groups):
        grouped_ids[group].append(id_)

    # Determine the number of samples per group
    if n_samples_per_group is not None:
        # Randomly sample from each group
        sampled_ids = []
        sampled_groups = []
        random.seed(random_state)  # For reproducibility
        for group, id_list in grouped_ids.items():
            selected = random.sample(id_list, min(n_samples_per_group, len(id_list)))  # Ensure we don't sample more than available
            sampled_ids.extend(selected)
            sampled_groups.extend([group] * len(selected))  # Keep track of groups
        ids = sampled_ids
        stratify_groups = sampled_groups


    # Initialize the Stratified K-Fold splitter
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    # List to hold the splits (train and test dictionaries)
    splits = []

    idx_split = 0

    print()
    print()
    print(f"{'=' * 74}")
    print(f"{'=' * 20} {k}-FOLD TRAINING/VALIDATION SPLIT {'=' * 20}")
    print(f"{'=' * 74}")
    print()

    # Perform the split based on IDs and stratified groups
    for train_idx, test_idx in skf.split(ids, stratify_groups):
        # Create dictionaries to hold train and test files in the same format as metadata
        train_files = {ids[i]: data_dictionary[ids[i]] for i in train_idx}
        test_files = {ids[i]: data_dictionary[ids[i]] for i in test_idx}

        print()
        print(f"{'=' * 20} SPLIT: {idx_split} {'=' * 20}")
        print()
    
        print(f"{'=' * 10}> TRAINING SET")
        print()
    
        print(f" SAMPLES: {train_files.keys()}")
        print()
    
        print(f"{'=' * 10}> VALIDATION SET")
        print()
    
        print(f" SAMPLES: {test_files.keys()}")

        # Append the tuple (train_files, test_files) as dictionaries
        splits.append((train_files, test_files))

        idx_split += 1

    return splits

def custom_collate(batch):
    # Collect x (dictionaries) into a list
    batch_sample_ids = [item[0] for item in batch]
    # Collect x (dictionaries) into a list
    batch_x = [item[1] for item in batch]
    # Collect y (strings) into a list
    batch_y = [item[2] for item in batch]

    return batch_sample_ids, batch_x, batch_y