import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nilearn import image
from nilearn.masking import apply_mask
from sklearn.preprocessing import minmax_scale

from utils.nifti_ops import get_nifti_images, divide_brain_mask, save_as_nifti


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

    for mask_idx, mask in enumerate(parcellated_masks):
        import time
        t  = time.time()
        
        # Get voxel coordinates where mask is True using np.nonzero
        voxel_indices = np.nonzero(mask)
        print(f"Mask {mask_idx} with {len(voxel_indices[0])} voxels.")

        voxel_values = np.zeros((len(accuracy_maps), len(voxel_indices[0])))

        # Load accuracy maps and extract voxel values
        for map_idx, accuracy_map in enumerate(accuracy_maps):
            accuracy_map_data = nib.load(accuracy_map).get_fdata()
            voxel_values[map_idx, :] = accuracy_map_data[voxel_indices]

        # Calculate the threshold statistic for each voxel using np.percentile
        percentile_value = 100 * (1 - p_value_threshold)
        threshold_statistic = np.percentile(voxel_values, percentile_value, axis=0)

        # Assign the computed threshold statistic to the map at the voxel indices
        threshold_statistics_map[voxel_indices] = threshold_statistic

        import psutil
        # Get memory usage in bytes after processing each mask
        memory_usage_bytes = psutil.Process().memory_info().rss
        print(f"Memory usage: {memory_usage_bytes/1024**2} MB")
        print(f"Time for parcellation: {time.time()-t} s")

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


def thresholded_cluster_sizes(
    map_index,
    accuracy_map_path,
    threshold_map_path,
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
    - threshold_map (str): Path to map of threshold values based on empirical chance distribution.
    - brain_mask_path (str): Path to the brain mask Nifti image.
    - output_dir (str): Directory path to store the cluster sizes.
    - connectivity (int): Connectivity scheme (6 or 18) for voxel connections.
    """
    accuracy_map_data = nib.load(str(accuracy_map_path)).get_fdata()
    brain_mask_data = nib.load(str(brain_mask_path)).get_fdata()
    threshold_map_data = nib.load(str(threshold_map_path)).get_fdata()

    cluster_map, cluster_sizes = clusters_above_threshold(
        accuracy_map_data, threshold_map_data, brain_mask_data
    )

    # Save cluster sizes to disk (or aggregate results)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"cluster_sizes_map_{map_index:06d}.json"
    output_path = Path(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(cluster_sizes, f)

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
    - cluster_sizes_original (Dict): Dictionary with cluster label key and size as value for the original data.

    Returns:
    - cluster_p_values (dict): Dictionary mapping cluster sizes to computed p-values.
    """
    cs_chance = np.array(cluster_sizes_chance)
    cluster_p_values = {}
    for cl_label, cl_size in cluster_sizes_original.items():
        # Compute cluster-level p-value
        p_cluster = np.sum(cs_chance > cl_size) / len(cluster_sizes_chance)
        cluster_p_values[cl_size] = p_cluster

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
    accuracy_maps: List[Path],
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
    filename = f"voxel_wise_threshold_map_{p_value_threshold}.nii.gz"
    fpath = Path(output_path, filename)
    save_as_nifti(fpath, mask_path, threshold_statistics_maps)
    return fpath


def prepare_cluster_statistics(
    accuracy_maps: List[Path], threshold_map_path: Path, brain_mask_path: Path, output_dir: Path
):
    # Create a pool of worker processes
    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)

    parallel = True
    
    # Process accuracy maps in parallel
    results = []
    for idx, accuracy_map_file in enumerate(accuracy_maps):
        if parallel:
            result = pool.apply_async(
                thresholded_cluster_sizes,
                args=(
                    idx,
                    accuracy_map_file,
                    threshold_map_path,
                    brain_mask_path,
                    output_dir,
                ),
            )
            results.append(result.get()[1])
        else:
            _, result = thresholded_cluster_sizes(
                idx,
                accuracy_map_file,
                threshold_map_path,
                brain_mask_path,
                output_dir,
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
    
    # Store the data back to the image
    map_img = nib.Nifti1Image(map_data, map_img.affine)

    # Apply cluster size threshold
    thresh_img = image.threshold_img(map_img, 0.1, cluster_size)
    return thresh_img


if __name__ == "__main__":
    permutation_dir = Path("/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/random_accuracy_maps")
    brain_mask = Path("/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/brain_mask.nii.gz")
    # This represents the true original accuracy map. Here we use a map generated by generate_fake_clusters
    # This map represents the group accuracy map, that is the mean of the single-subject accuracy maps
    accuracy_map = Path("/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/permutations/clustered_accuracy_map.nii.gz")
    base_output_path = Path("/Users/sebastian.hoefle/projects/idor/brain-analysis/fixtures/stat_results")
    perm_output_path = Path(base_output_path, "permuted")
    original_output_path = Path(base_output_path, "original")

    # Fig 1.C: Use voxel-wise distribution of permuted accuracy maps to determine the threshold map
    accuracy_maps = get_nifti_images(permutation_dir)
    n_parcellations = 10  # Use higher values if you face memory issues
    voxel_wise_p_value = 0.001
    threshold_map = Path(base_output_path, f"voxel_wise_threshold_map_{voxel_wise_p_value}.nii.gz")
    if not threshold_map.exists():
        threshold_map = calculate_voxel_wise_threshold_map(
            accuracy_maps, brain_mask, base_output_path, voxel_wise_p_value, n_parcellations
        )

    
    # Fig 1. D: Process all permutation maps to determine the cluster sizes given by chance
    cluster_sizes_chance = prepare_cluster_statistics(
        accuracy_maps, threshold_map, brain_mask, perm_output_path
    )

    # Determine the cluster map for the true original map
    map_idx = 0
    cluster_map_original, cluster_sizes_original = thresholded_cluster_sizes(
        map_idx, accuracy_map, threshold_map, brain_mask, original_output_path
    )
    cluster_map_path = Path(base_output_path, "original", f"cluster_map_{map_idx:06d}.nii.gz")
    
    # cluster_sizes_chance is a list of cluster sizes for each permutation map.
    # The cluster_sizes per permutation map is a dictionary where each key represents a cluster label and its corresponding value represents the size of that cluster.
    sz_chance_nested = [list(sz.values()) for sz in cluster_sizes_chance]
    sz_chance = [item for sublist in sz_chance_nested for item in sublist]
    # Determine significant cluster size with FDR correction
    significant_cluster_size = cluster_threshold_stats(sz_chance, cluster_sizes_original, alpha=0.05)
    
    # Final step
    image = keep_significant_clusters(accuracy_map, cluster_map_path, significant_cluster_size)
    nib.save(image, str(Path(original_output_path, "image_cluster_FDR_corrected.nii.gz")))
