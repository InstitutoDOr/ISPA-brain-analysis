import os
import numpy as np
from nilearn import image
from nilearn.masking import apply_mask


import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from sklearn.preprocessing import minmax_scale


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
        parcellation_mask[
            nonzero_indices[0][start_idx:end_idx],
            nonzero_indices[1][start_idx:end_idx],
            nonzero_indices[2][start_idx:end_idx],
        ] = True

        # Append the parcellation mask to the list
        parcellations.append(parcellation_mask)

    return parcellations


import os
import numpy as np
import nibabel as nib


def calculate_threshold_statistics_maps(
    image_dir, parcellated_masks, p_value_threshold=0.001
):
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
                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                    accuracy_map_path = os.path.join(image_dir, filename)
                    accuracy_map_img = nib.load(accuracy_map_path)
                    accuracy_map_data = accuracy_map_img.get_fdata()
                    # Find non-zero indices in the mask
                    nonzero_indices = np.nonzero(mask_data)

                    # Get voxel value from accuracy map at current voxel index
                    voxel_value = accuracy_map_data[voxel_idx]
                    voxel_values.append(voxel_value)

            # Calculate the threshold statistic using np.percentile
            threshold_statistic = np.percentile(
                voxel_values, 100 * (1 - p_value_threshold)
            )

            # Store the threshold statistic in the map at the current voxel index
            threshold_statistics_map[voxel_idx] = threshold_statistic

    return threshold_statistics_map


def store_threshold_statistics_to_nifti(
    output_path, brain_mask_path, threshold_statistics_map
):
    """
    Store threshold statistics as a Nifti image using the brain mask.

    Parameters:
    - output_path (str): Path to save the output Nifti image.
    - brain_mask_path (str): Path to the brain mask Nifti image.
    - threshold_statistics_map (ndarray): Voxel-wise threshold statistics map.
    """
    brain_mask_img = nib.load(brain_mask_path)

    # Create a new Nifti image using the brain mask
    threshold_img = nib.Nifti1Image(
        threshold_statistics_map, brain_mask_img.affine, brain_mask_img.header
    )

    # Save the threshold statistics as a Nifti image
    nib.save(threshold_img, output_path)


def perform_cluster_search_with_size(
    accuracy_map, threshold_map, brain_mask, connectivity=6
):
    """
    Perform cluster search and collect cluster sizes using a predefined threshold map within a specified brain mask.

    Parameters:
    - accuracy_map (ndarray): 3D array representing accuracy values at each voxel.
    - threshold_map (ndarray): 3D array representing threshold values based on empirical chance distribution.
    - brain_mask (ndarray): 3D array specifying voxels to consider (1 for inclusion, 0 for exclusion).
    - connectivity (int): Connectivity scheme (6 or 18) for voxel connections.

    Returns:
    - cluster_map (ndarray): 3D array indicating clusters identified based on threshold and connectivity within the brain mask.
    - cluster_sizes (dict): Dictionary mapping cluster labels to their sizes.
    """
    cluster_map = np.zeros_like(accuracy_map, dtype=int)
    cluster_sizes = {}

    if connectivity == 6:
        structuring_element = ndi.generate_binary_structure(3, 1)  # 6-connectivity
    elif connectivity == 18:
        structuring_element = ndi.generate_binary_structure(3, 2)  # 18-connectivity

    cluster_label = 1

    for x in range(accuracy_map.shape[0]):
        for y in range(accuracy_map.shape[1]):
            for z in range(accuracy_map.shape[2]):
                if (
                    brain_mask[x, y, z] == 1  # Check if voxel is within the brain mask
                    and accuracy_map[x, y, z] > threshold_map[x, y, z]
                    and cluster_map[x, y, z] == 0
                ):
                    # Start a new cluster
                    cluster_map[x, y, z] = cluster_label
                    queue = [(x, y, z)]
                    cluster_size = 0

                    while queue:
                        vx, vy, vz = queue.pop(0)
                        cluster_size += 1

                        for dx, dy, dz in np.transpose(np.nonzero(structuring_element)):
                            nx, ny, nz = vx + dx, vy + dy, vz + dz
                            if (
                                0 <= nx < accuracy_map.shape[0]
                                and 0 <= ny < accuracy_map.shape[1]
                                and 0 <= nz < accuracy_map.shape[2]
                                and brain_mask[nx, ny, nz]
                                == 1  # Check if neighboring voxel is within the brain mask
                                and accuracy_map[nx, ny, nz] > threshold_map[nx, ny, nz]
                                and cluster_map[nx, ny, nz] == 0
                            ):
                                cluster_map[nx, ny, nz] = cluster_label
                                queue.append((nx, ny, nz))

                    # Store the cluster size
                    cluster_sizes[cluster_label] = cluster_size
                    cluster_label += 1

    return cluster_map, cluster_sizes


def collect_cluster_sizes(
    accuracy_maps, threshold_map, p_value_threshold=0.001, connectivity=6
):
    """
    Apply cluster search to multiple accuracy maps and collect cluster sizes.

    Parameters:
    - accuracy_maps (list of ndarray): List of 3D arrays representing accuracy maps.
    - threshold_map (ndarray): 3D array representing threshold values based on empirical chance distribution.
    - p_value_threshold (float): Desired p-value threshold for including voxels in clusters.
    - connectivity (int): Connectivity scheme (6 or 18) for voxel connections.

    Returns:
    - all_cluster_sizes (list of dict): List of dictionaries mapping cluster labels to their sizes for each accuracy map.
    """
    all_cluster_sizes = []

    for accuracy_map in accuracy_maps:
        _, cluster_sizes = perform_cluster_search_with_size(
            accuracy_map, threshold_map, p_value_threshold, connectivity
        )
        all_cluster_sizes.append(cluster_sizes)

    return all_cluster_sizes


def process_accuracy_map(
    map_index, accuracy_map_path, threshold_map, brain_mask_path, connectivity=6
):
    """
    Process a single accuracy map and save cluster sizes to disk.

    Parameters:
    - map_index (int): Index of the accuracy map.
    - accuracy_map_path (str): Path to the accuracy map Nifti image.
    - threshold_map (ndarray): 3D array representing threshold values based on empirical chance distribution.
    - brain_mask_path (str): Path to the brain mask Nifti image.
    - connectivity (int): Connectivity scheme (6 or 18) for voxel connections.
    """
    accuracy_map_img = nib.load(accuracy_map_path)
    accuracy_map_data = accuracy_map_img.get_fdata()
    brain_mask_img = nib.load(brain_mask_path)
    brain_mask_data = brain_mask_img.get_fdata()

    cluster_sizes = perform_cluster_search(
        accuracy_map_data, threshold_map, brain_mask_data, connectivity
    )

    # Save cluster sizes to disk (or aggregate results)
    output_filename = f"cluster_sizes_map_{map_index:06d}.txt"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        for label, size in cluster_sizes.items():
            f.write(f"Cluster {label}: Size {size}\n")


def compute_cluster_p_values(cluster_sizes_chance, cluster_sizes_original):
    """
    Compute cluster-level p-values based on chance and original cluster size records.

    Parameters:
    - cluster_sizes_chance (list): List of cluster sizes from chance population.
    - cluster_sizes_original (list): List of cluster sizes from original data.

    Returns:
    - cluster_p_values (dict): Dictionary mapping cluster sizes to computed p-values.
    """
    # Compute normalized histogram (chance record)
    hist_chance, _ = np.histogram(cluster_sizes_chance, bins="auto", density=True)

    cluster_p_values = {}
    for s in cluster_sizes_original:
        # Compute cluster-level p-value
        p_cluster = np.sum(hist_chance[hist_chance > hist_chance[s]])
        cluster_p_values[s] = p_cluster

    return cluster_p_values


def apply_fdr_correction(cluster_p_values, alpha=0.05):
    """
    Apply step-down False Discovery Rate (FDR) correction to cluster p-values.

    Parameters:
    - cluster_p_values (dict): Dictionary mapping cluster sizes to computed p-values.
    - alpha (float): Desired FDR threshold (e.g., 0.05).

    Returns:
    - cluster_size_threshold (float): Adjusted cluster size threshold based on FDR correction.
    """
    sorted_p_values = sorted(cluster_p_values.values())
    m = len(sorted_p_values)  # Number of comparisons (clusters)
    adjusted_p_values = [p * m / (i + 1) for i, p in enumerate(sorted_p_values)]

    # Find largest adjusted p-value <= alpha
    for i in range(m - 1, -1, -1):
        if adjusted_p_values[i] <= alpha:
            cluster_size_threshold = list(cluster_p_values.keys())[i]
            break

    return cluster_size_threshold


def apply_cluster_threshold(
    original_accuracy_map, threshold_map, cluster_size_threshold
):
    """
    Apply cluster size threshold to original accuracy map.

    Parameters:
    - original_accuracy_map (ndarray): 3D array representing original accuracy values.
    - threshold_map (ndarray): 3D array representing threshold values based on empirical chance distribution.
    - cluster_size_threshold (float): Cluster size threshold for significance.

    Returns:
    - filtered_accuracy_map (ndarray): Filtered accuracy map based on cluster size threshold.
    """
    # Apply cluster size threshold to filter clusters
    filtered_clusters = original_accuracy_map >= cluster_size_threshold

    # Apply voxel-wise p-values to retain significant clusters
    filtered_accuracy_map = original_accuracy_map * filtered_clusters

    return filtered_accuracy_map


def process_cluster_thresholds():
    # Example usage (to be integrated with your workflow)
    cluster_sizes_chance = [...]  # List of cluster sizes from chance population
    cluster_sizes_original = [...]  # List of cluster sizes from original data

    # Compute cluster-level p-values
    cluster_p_values = compute_cluster_p_values(
        cluster_sizes_chance, cluster_sizes_original
    )

    # Apply FDR correction to determine cluster size threshold
    cluster_size_threshold = apply_fdr_correction(cluster_p_values)

    # Example: Load original accuracy map and threshold map
    original_accuracy_map = np.load("original_accuracy_map.npy")
    threshold_map = np.load("threshold_map.npy")

    # Apply cluster threshold to filter accuracy map
    filtered_accuracy_map = apply_cluster_threshold(
        original_accuracy_map, threshold_map, cluster_size_threshold
    )

    # Example: Save filtered accuracy map
    np.save("filtered_accuracy_map.npy", filtered_accuracy_map)


def calculate_voxel_wise_threshold_maps(
    image_dir: Path, mask_path: Path, output_path: Path
):
    n_parcellations = 10  # Adapt this number according to you available memory. More parcellations need less memory

    # Divide the brain mask into parcellations
    parcellations = divide_brain_mask(mask_path, n_parcellations)

    # Save each parcellation mask as a Nifti file
    output_dir = "/Users/sebastian.hoefle/projects/idor/brain-analysis/Data/"
    os.makedirs(output_dir, exist_ok=True)

    for i, parcellation_mask in enumerate(parcellations):
        parcellation_img = nib.Nifti1Image(
            parcellation_mask.astype(np.uint8), nib.load(mask_path).affine
        )
        output_filename = f"parcellation_{i + 1}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        nib.save(parcellation_img, output_path)
        print(f"Parcellation mask {i + 1} saved to: {output_path}")

    # Calculate threshold statistics maps for each parcellation
    threshold_statistics_maps = calculate_threshold_statistics_maps(
        image_dir, parcellations
    )

    # Store combined threshold statistics map as a Nifti image
    output_path = "threshold_statistics_map.nii.gz"
    store_threshold_statistics_to_nifti(
        output_path, mask_path, threshold_statistics_maps
    )


def calculate_cluster_statistics(permutation_dir: Path, threshold_map: Path, brain_mask: Path, output_dir: Path):
    accuracy_maps_dir = "/path/to/accuracy_maps"
    threshold_map_path = "/path/to/threshold_map.nii.gz"
    brain_mask_path = "/path/to/brain_mask.nii.gz"
    output_dir = "/path/to/output_directory"

    # List all accuracy map files
    accuracy_map_files = [
        os.path.join(permutation_dir, f)
        for f in os.listdir(permutation_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    # Load threshold map and brain mask
    threshold_map_img = nib.load(str(threshold_map_path))
    threshold_map_data = threshold_map_img.get_fdata()
    brain_mask_img = nib.load(str(brain_mask_path))
    brain_mask_data = brain_mask_img.get_fdata()

    # Define connectivity scheme (6 or 18)
    connectivity = 6

    # Create a pool of worker processes
    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)

    # Process accuracy maps in parallel
    results = []
    for idx, accuracy_map_file in enumerate(accuracy_map_files):
        result = pool.apply_async(
            process_accuracy_map,
            args=(
                idx,
                accuracy_map_file,
                threshold_map_data,
                brain_mask_path,
                connectivity,
            ),
        )
        results.append(result)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    print("All accuracy maps processed.")


if __name__ == "__main__":
    permutation_dir = "/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/random_accuracy_maps"
    brain_mask = "/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/brain_mask.nii.gz"
    accuracy_map = "/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/clustered_accuracy_map.nii.gz"

    # TODO: organize the above methods that they get the correct path as parameters and store the respective output to directories
    # TODO: Test the statistics with the sample files created wit `test_permutation_maps.py`
