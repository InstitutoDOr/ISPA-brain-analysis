import os

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nilearn import image
from nilearn.masking import apply_mask
from sklearn.preprocessing import minmax_scale


def calculate_threshold_statistics_map(
    accuracy_maps: list, parcellated_masks: list, p_value_threshold: float
):
    """
    Calculate voxel-wise threshold statistics corresponding to a low probability across multiple maps and parcellated masks.

    Reference:
    Johannes Stelzer, Yi Chen, Robert Turner,
    Statistical inference and multiple testing correction in classification-based multi-voxel pattern analysis (MVPA): Random permutations and cluster size control, NeuroImage, Volume 65, 2013, Pages 69-82, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2012.09.063.

    Parameters:
    - accuracy_maps (list of paths): Filepaths of Nifti accuracy.
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
            for accuracy_map_path in accuracy_maps:
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


def clusters_above_threshold(accuracy_map, threshold_map, brain_mask):
    """
    Collect cluster sizes using a predefined threshold map within a specified brain mask.

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

    connectivity = 6
    if connectivity == 6:
        structuring_element = ndi.generate_binary_structure(3, 1)  # 6-connectivity
    else:
        raise NotImplementedError(
            "nilearn uses hard-coded 6-connectivity. Not implemented for compatibility. ",
            "Ref: https://github.com/nilearn/nilearn/blob/4f4730163097457cf9ddb5674ffd158ee8fa822e/nilearn/image/image.py#L847-L848",
        )
    # elif connectivity == 18:
    #     structuring_element = ndi.generate_binary_structure(3, 2)  # 18-connectivity

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

                        # Loop over connected voxels and extend the current cluster
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


def collect_cluster_sizes(accuracy_maps, threshold_map, brain_mask_data):
    """
    Apply cluster search to multiple accuracy maps and collect cluster sizes.

    Parameters:
    - accuracy_maps (list of ndarray): List of 3D arrays representing accuracy maps.
    - threshold_map (ndarray): 3D array representing threshold values based on empirical chance distribution.

    Returns:
    - all_cluster_sizes (list of dict): List of dictionaries mapping cluster labels to their sizes for each accuracy map.
    """
    all_cluster_sizes = []

    for accuracy_map in accuracy_maps:
        _, cluster_sizes = clusters_above_threshold(accuracy_map, threshold_map, brain_mask_data)
        all_cluster_sizes.append(cluster_sizes)

    return all_cluster_sizes


def thresholded_cluster_sizes(
    map_index,
    accuracy_map_path,
    threshold_map,
    brain_mask_path,
    output_dir,
    connectivity=6,
    save_thresholded_map: bool = True,
):
    """
    Process a single accuracy map and save cluster sizes to disk.

    Parameters:
    - map_index (int): Index of the accuracy map.
    - accuracy_map_path (str): Path to the accuracy map Nifti image.
    - threshold_map (ndarray): 3D array representing threshold values based on empirical chance distribution.
    - brain_mask_path (str): Path to the brain mask Nifti image.
    - output_dir (str): Directory path to store the cluster sizes.
    - connectivity (int): Connectivity scheme (6 or 18) for voxel connections.
    """
    accuracy_map_img = nib.load(accuracy_map_path)
    accuracy_map_data = accuracy_map_img.get_fdata()
    brain_mask_img = nib.load(brain_mask_path)
    brain_mask_data = brain_mask_img.get_fdata()

    cluster_map, cluster_sizes = clusters_above_threshold(
        accuracy_map_data, threshold_map, brain_mask_data
    )

    # Save cluster sizes to disk (or aggregate results)
    output_filename = f"cluster_sizes_map_{map_index:06d}.txt"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        for label, size in cluster_sizes.items():
            f.write(f"Cluster {label}: Size {size}\n")

    # Save cluster map to disk (or aggregate results)
    if save_thresholded_map:
        output_filename = f"cluster_map_{map_index:06d}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        save_as_nifti(output_path, brain_mask_path, cluster_map)
    return cluster_map, cluster_sizes


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


def cluster_threshold_stats(cluster_sizes_chance, cluster_sizes_original, alpha=0.05):
    # Compute cluster-level p-values
    cluster_p_values = compute_cluster_p_values(
        cluster_sizes_chance, cluster_sizes_original
    )

    # Apply FDR correction to determine cluster size threshold
    cluster_size_threshold = apply_fdr_correction(cluster_p_values, alpha)

    return cluster_size_threshold

    # Example: Load original accuracy map and threshold map
    original_accuracy_map = np.load("original_accuracy_map.npy")
    threshold_map = np.load("threshold_map.npy")

    # Apply cluster threshold to filter accuracy map
    filtered_accuracy_map = apply_cluster_threshold(
        original_accuracy_map, threshold_map, cluster_size_threshold
    )

    # Example: Save filtered accuracy map
    np.save("filtered_accuracy_map.npy", filtered_accuracy_map)


def calculate_voxel_wise_threshold_map(
    accuracy_maps: list[Path],
    mask_path: Path,
    output_path: Path,
    p_value_threshold: float = 0.001,
    n_parcellations: int = 10,
):
    """Uses voxel-wise distribution to determine the voxel-wise threshold map

    This corresponds to part C in Fig. 1 from Stelzer et. al 2013.

    Args:
        accuracy_maps (list): Filepaths of accuracy maps
        mask_path (Path): Path of brain mask.
        output_path (Path): Directory to store the mask parcellation and the voxel-wise threshold map
        p_value_threshold: p-value to use to determine the threshold. Defaults to 0.001
    """
    # Divide the brain mask into parcellations
    parcellations = divide_brain_mask(mask_path, n_parcellations)

    # Save each parcellation mask as a Nifti file
    output_path.mkdir(parents=True, exist_ok=True)

    # Optional: Store the parcellation to disk
    for i, parcellation_mask in enumerate(parcellations):
        parcellation_img = nib.Nifti1Image(
            parcellation_mask.astype(np.uint8), nib.load(mask_path).affine
        )
        output_filename = f"parcellation_{i + 1}.nii.gz"
        filepath = Path(output_path, output_filename)
        nib.save(parcellation_img, filepath)
        print(f"Parcellation mask {i + 1} saved to: {filepath}")

    # Calculate threshold statistics maps for each parcellation
    threshold_statistics_maps = calculate_threshold_statistics_map(
        accuracy_maps, parcellations, p_value_threshold
    )

    # Store combined threshold statistics map as a Nifti image
    filename = "voxel_wise_threshold_map_{p_value_threshold}.nii.gz"
    fpath = Path(output_path, filename)
    save_as_nifti(fpath, mask_path, threshold_statistics_maps)
    return fpath


def prepare_cluster_statistics(
    accuracy_maps: list[Path], threshold_map: Path, brain_mask: Path, output_dir: Path
):
    # Load threshold map and brain mask
    threshold_map_img = nib.load(str(threshold_map))
    threshold_map_data = threshold_map_img.get_fdata()
    brain_mask_img = nib.load(str(brain_mask))
    brain_mask_data = brain_mask_img.get_fdata()

    # Define connectivity scheme (6 or 18)
    connectivity = 6

    # Create a pool of worker processes
    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)

    # Process accuracy maps in parallel
    results = []
    for idx, accuracy_map_file in enumerate(accuracy_map_files):
        _, result = pool.apply_async(
            thresholded_cluster_sizes,
            args=(
                idx,
                accuracy_map_file,
                threshold_map_data,
                brain_mask_path,
                output_dir,
                connectivity,
            ),
        )
        results.append(result)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    print("All cluster sizes determined.")
    return results


def keep_significant_clusters(accuracy_map, accuracy_cluster_map, cluster_size):
    # Original accuracy map, unthresholded
    map_img = nib.load(str(accuracy_map))
    map_data = map_img.get_fdata()
    
    # Use the cluster map that contains non-zero cluster labels for the voxel-wise thresholded map
    cluster_map = nib.load(str(accuracy_cluster_map)).get_fdata()
    
    # Zero all voxels not are not part of a cluster
    map_data[cluster_map < 1] = 0 
    
    # Apply cluster size threshold
    thresh_img = nilearn.image.threshold_img(map_img, 0.1, cluster_size)
    return thresh_img


if __name__ == "__main__":
    permutation_dir = "/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/random_accuracy_maps"
    brain_mask = "/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/brain_mask.nii.gz"
    accuracy_map = "/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/clustered_accuracy_map.nii.gz"
    base_output_path = "/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/stat_results"

    # TODO: organize the above methods that they get the correct path as parameters and store the respective output to directories
    accuracy_maps = get_nifti_images(permutation_dir)
    n_parcellations = 10  # Use higher values if you face memory issues
    voxel_wise_p_value = 0.001
    threshold_map = calculate_voxel_wise_threshold_map(
        accuracy_maps, brain_mask, output_path, voxel_wise_p_value, n_parcellations
    )

    # Process all permutation maps to determine the cluster sizes given by chance
    perm_output_path = Path(base_output_path, "permuted")
    cluster_sizes_chance = prepare_cluster_statistics(
        accuracy_maps, threshold_map, brain_mask, perm_output_path
    )

    # Determine the cluster map for the true original map
    original_output_path = Path(base_output_path, "original")
    map_idx = 0
    cluster_map_original, cluster_sizes_original = thresholded_cluster_sizes(
        map_idx, accuracy_map, threshold_map, brain_mask_data, original_output_path
    )
    cluster_map_path = Path(base_output_path, "original", f"cluster_map_{map_idx:06d}.nii.gz")
    
    # Determine significant cluster size with FDR correction
    significant_cluster_size = cluster_threshold_stats(
        cluster_sizes_chance, cluster_sizes_original, alpha=0.05
    )
    
    # Final step
    image = keep_significant_clusters(accuracy_map, cluster_map_path, significant_cluster_size)
    nib.save(image, str(Path(original_output_path, "image_cluster_FDR_corrected.nii.gz")))
