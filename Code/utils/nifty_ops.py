from pathlib import Path

import nibabel as nib


def divide_brain_mask(mask_path: Path, n_parcellations: int):
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
        parcellation_mask[
            nonzero_indices[0][start_idx:end_idx],
            nonzero_indices[1][start_idx:end_idx],
            nonzero_indices[2][start_idx:end_idx],
        ] = True

        # Append the parcellation mask to the list
        parcellations.append(parcellation_mask)

    return parcellations


def get_nifti_images(image_dir, prefix: str = "", absolute: bool = True):
    files = os.listdir(image_dir)
    files = [f for f in files if f.endswith(".nii") or f.endswith(".nii.gz")]
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    if absolute:
        files = [os.path.join(image_dir, f) for f in files]
    return files


def save_as_nifti(output_path: Path, brain_mask_path: Path, data: ndarray):
    """
    Store threshold statistics as a Nifti image using the brain mask.

    Parameters:
    - output_path (str): Path to save the output Nifti image.
    - brain_mask_path (str): Path to the brain mask Nifti image.
    - data (ndarray): Nifti data that needs to match the dimensions of the brain mask.
    """
    mask_img = nib.load(brain_mask_path)

    # Create a new Nifti image using the brain mask
    threshold_img = nib.Nifti1Image(data, mask_img.affine, mask_img.header)

    # Save the threshold statistics as a Nifti image
    nib.save(threshold_img, str(output_path))
