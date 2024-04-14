import os
import numpy as np
from nilearn import image
from nilearn.masking import apply_mask


import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import minmax_scale


import numpy as np
import nibabel as nib

import numpy as np
import nibabel as nib

import numpy as np
import nibabel as nib

def divide_brain_mask(mask_path, n_parcellations):
    """
    Divdes a brain mask into parcellations using only on non-zero elements.

    Parameters:
    - mask_path (str): Path to the brain mask Nifti image.
    - n_parcellations (int): Number of parcellations to divide the mask into.

    Returns:
    - parcellations (list of ndarray): List of parcellated masks (arrays of voxel indices).
    """

    # Load the brain mask
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # Find non-zero indices in the mask
    nonzero_indices = np.nonzero(mask_data)

    # Get the number of non-zero voxels
    num_nonzero_voxels = len(nonzero_indices[0])

    # Calculate the approximate size of each parcellation
    parcellation_size = num_nonzero_voxels // n_parcellations

    # Initialize list to store parcellation masks
    parcellations = []

    # Create parcellations based on non-zero voxel indices
    for i in range(n_parcellations):
        start_idx = i * parcellation_size
        end_idx = (i + 1) * parcellation_size
        if i == n_parcellations - 1:
            end_idx = num_nonzero_voxels  # Extend to the end for the last parcellation

        # Create a new mask array for the current parcellation
        parcellation_mask = np.zeros_like(mask_data, dtype=bool)

        # Set the parcellation voxels to True based on their indices
        parcellation_mask[nonzero_indices[0][start_idx:end_idx],
                          nonzero_indices[1][start_idx:end_idx],
                          nonzero_indices[2][start_idx:end_idx]] = True

        # Append the parcellation mask to the list
        parcellations.append(parcellation_mask)

    return parcellations


import os
import numpy as np
import nibabel as nib

def calculate_threshold_statistics_maps(image_dir, parcellated_masks, p_value_threshold=0.001):
    """
    Calculate voxel-wise threshold statistics corresponding to a low probability across multiple maps and parcellated masks.

    Reference:
    Johannes Stelzer, Yi Chen, Robert Turner,
    Statistical inference and multiple testing correction in classification-based multi-voxel pattern analysis (MVPA): Random permutations and cluster size control, NeuroImage, Volume 65, 2013, Pages 69-82, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2012.09.063.

    Parameters:
    - image_dir (str): Directory containing accuracy map Nifti images.
    - parcellated_masks (list of ndarray): List of parcellated masks (arrays of voxel indices).
    - p_value_threshold (float): Desired probability threshold.

    Returns:
    - threshold_statistics_map (ndarray): Voxel-wise threshold statistics map.
    """
    # Initialize an empty map to store threshold statistics
    threshold_statistics_map = np.zeros_like(parcellated_masks[0], dtype=float)

    # Iterate over each parcellated mask
    for mask_idx, mask in enumerate(parcellated_masks):
        # Get voxel coordinates where mask is True using np.nonzero
        voxel_indices = np.nonzero(mask)

        # Iterate over voxel coordinates
        for voxel_idx in zip(*voxel_indices):
            voxel_values = []

            # Load accuracy maps from image directory
            for filename in os.listdir(image_dir):
                if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                    accuracy_map_path = os.path.join(image_dir, filename)
                    accuracy_map_img = nib.load(accuracy_map_path)
                    accuracy_map_data = accuracy_map_img.get_fdata()

                    # Get voxel value from accuracy map at current voxel index
                    voxel_value = accuracy_map_data[voxel_idx]
                    voxel_values.append(voxel_value)

            # Calculate the threshold statistic using np.percentile
            threshold_statistic = np.percentile(voxel_values, 100 * (1 - p_value_threshold))

            # Store the threshold statistic in the map at the current voxel index
            threshold_statistics_map[voxel_idx] = threshold_statistic

    return threshold_statistics_map


def store_threshold_statistics_to_nifti(output_path, brain_mask_path, threshold_statistics_map):
    """
    Store threshold statistics as a Nifti image using the brain mask.

    Parameters:
    - output_path (str): Path to save the output Nifti image.
    - brain_mask_path (str): Path to the brain mask Nifti image.
    - threshold_statistics_map (ndarray): Voxel-wise threshold statistics map.
    """
    brain_mask_img = nib.load(brain_mask_path)

    # Create a new Nifti image using the brain mask
    threshold_img = nib.Nifti1Image(threshold_statistics_map, brain_mask_img.affine, brain_mask_img.header)

    # Save the threshold statistics as a Nifti image
    nib.save(threshold_img, output_path)

# Example usage
mask_path = '/Users/sebastian.hoefle/projects/idor/brain-analysis/Data/brain_mask.nii.gz'
image_dir = '/Users/sebastian.hoefle/projects/idor/brain-analysis/Data/ISPA_SearchLight_TB_permut_test/withinsubj_permut_test/permu_mean_group_acc_maps_test'
n_parcellations = 10  # Adapt this number according to you available memory. More parcellations need less memory

# Divide the brain mask into parcellations
parcellations = divide_brain_mask(mask_path, n_parcellations)

# Save each parcellation mask as a Nifti file
output_dir = "/Users/sebastian.hoefle/projects/idor/brain-analysis/Data/"
os.makedirs(output_dir, exist_ok=True)

for i, parcellation_mask in enumerate(parcellations):
    parcellation_img = nib.Nifti1Image(parcellation_mask.astype(np.uint8), nib.load(mask_path).affine)
    output_filename = f"parcellation_{i + 1}.nii.gz"
    output_path = os.path.join(output_dir, output_filename)
    nib.save(parcellation_img, output_path)
    print(f"Parcellation mask {i + 1} saved to: {output_path}")

# Calculate threshold statistics maps for each parcellation
threshold_statistics_maps = calculate_threshold_statistics_maps(image_dir, parcellations)

# Store combined threshold statistics map as a Nifti image
output_path = '/path/to/threshold_statistics_map.nii.gz'
store_threshold_statistics_to_nifti(output_path, mask_path, threshold_statistics_maps)
