from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img, new_img_like


def create_sphere_mask(s_x, s_y, s_z, radius):
    """
    Create a sphere-shaped mask inside a 3D volume with specified dimensions.

    Parameters:
    - s_x (int): Size of the volume along the x-axis.
    - s_y (int): Size of the volume along the y-axis.
    - s_z (int): Size of the volume along the z-axis.
    - radius (float): Radius of the sphere.

    Returns:
    - brain_mask_img (Nifti1Image): Nifti image representing the sphere-shaped brain mask.
    """
    # Create a 3D grid of coordinates
    x, y, z = np.ogrid[-s_x // 2 : s_x // 2, -s_y // 2 : s_y // 2, -s_z // 2 : s_z // 2]

    # Compute the distance from the center of the volume
    distance = np.sqrt(x**2 + y**2 + z**2)

    # Create a binary mask where values <= radius are True (inside the sphere), otherwise False
    sphere_mask = distance <= radius

    # Create an empty 3D volume array
    volume_data = np.zeros((s_x, s_y, s_z))

    # Place the sphere mask at the center of the volume
    start_x = (s_x - sphere_mask.shape[0]) // 2
    start_y = (s_y - sphere_mask.shape[1]) // 2
    start_z = (s_z - sphere_mask.shape[2]) // 2
    volume_data[
        start_x : start_x + sphere_mask.shape[0],
        start_y : start_y + sphere_mask.shape[1],
        start_z : start_z + sphere_mask.shape[2],
    ] = sphere_mask.astype(float)

    # Create a Nifti image from the volume data
    brain_mask_img = nib.Nifti1Image(volume_data, affine=np.eye(4))

    return brain_mask_img


def generate_sphere_mask(output_directory: Path):
    # Example usage:
    s_x, s_y, s_z = 18, 20, 22  # Dimensions of the 3D volume
    radius = 8  # Radius of the sphere

    # Create the sphere-shaped brain mask
    brain_mask_img = create_sphere_mask(s_x, s_y, s_z, radius)

    # Save the brain mask as a Nifti file
    output_path = Path(output_directory, "brain_mask.nii.gz")
    nib.save(brain_mask_img, str(output_path))

    print(f"Sphere brain mask saved to: {output_path}")
    return output_path


def create_random_accuracy_maps(mask_path: Path, n_maps, mean=0.5, std=0.1):
    """
    Create random accuracy maps based on a brain mask from a Nifti file.

    Parameters:
    - mask_path (str): Path to the brain mask Nifti image.
    - n_maps (int): Number of random accuracy maps to generate.
    - mean (float): Mean of the Gaussian distribution.
    - std (float): Standard deviation of the Gaussian distribution.

    Returns:
    - accuracy_maps (list of Nifti1Image): List of random accuracy maps.
    """
    # Load the brain mask from the Nifti file
    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata().astype(bool)  # Convert mask data to boolean

    # Get the shape of the mask (same as accuracy maps)
    mask_shape = mask_data.shape

    # Initialize a list to store the random accuracy maps
    accuracy_maps = []

    # Generate random accuracy maps
    for _ in range(n_maps):
        # Generate random values from a Gaussian distribution
        random_values = np.random.normal(mean, std, mask_shape)

        # Clip values to ensure they are between 0 and 1
        random_values = np.clip(random_values, 0, 1)

        # Apply the brain mask to retain values only within the brain
        random_values[~mask_data] = 0  # Set values outside the mask to 0

        # Create a Nifti image from the random values
        accuracy_map_img = nib.Nifti1Image(random_values, mask_img.affine)

        # Append the accuracy map to the list
        accuracy_maps.append(accuracy_map_img)

    return accuracy_maps


def generate_and_save_random_accuracy_maps(
    mask_path: Path, output_dir: Path, n_maps: int, mean: float = 0.5, std: float = 0.1
):
    """
    Generate and save random accuracy maps based on a brain mask.

    Parameters:
    - mask_path (str): Path to the brain mask Nifti image.
    - output_dir (str): Directory to save the generated accuracy maps.
    - n_maps (int): Number of random accuracy maps to generate.
    - mean (float): Mean of the Gaussian distribution.
    - std (float): Standard deviation of the Gaussian distribution.
    """
    # Generate random accuracy maps
    accuracy_maps = create_random_accuracy_maps(mask_path, n_maps, mean, std)

    # Save the accuracy maps to disk
    for i, acc_map in enumerate(accuracy_maps):
        output_path = f"{output_dir}/random_accuracy_map_{i + 1:05d}.nii.gz"
        nib.save(acc_map, output_path)
        print(f"Saved random accuracy map {i + 1} to: {output_path}")


def generate_clustered_accuracy_map(
    mask_path, output_path, cluster_sizes, cluster_values, mean=0.5, std=0.001
):
    """
    Generate a Nifti file containing clusters of accuracy values above the chance level.

    Parameters:
    - mask_path (str): Path to the brain mask Nifti image.
    - output_path (str): Path to save the generated accuracy map.
    - cluster_sizes (list of int): List of cluster sizes.
    - cluster_values (list of float): List of accuracy values for each cluster.
    - mean (float): Mean of the Gaussian distribution for chance level.
    - std (float): Standard deviation of the Gaussian distribution for chance level.
    """
    # Load the brain mask from the Nifti file
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)  # Convert mask data to boolean

    # Initialize array for the accuracy map
    accuracy_map = np.random.normal(mean, std, mask_data.shape)
    
    # Remove voxels outside the mask
    accuracy_map[~mask_data] = 0

    # Clip accuracy map values to be within [0, 1]
    accuracy_map = np.clip(accuracy_map, 0, 1)

    # Generate clusters above the chance level
    for size, value in zip(cluster_sizes, cluster_values):
        # Find random coordinates within the brain mask
        indices = np.where(mask_data)
        num_voxels = len(indices[0])
        
        # Randomly select a central voxel index for the cluster
        central_idx = np.random.choice(num_voxels)
        central_x, central_y, central_z = indices[0][central_idx], indices[1][central_idx], indices[2][central_idx]

        # Create a spherical cluster around the central voxel
        for idx in range(num_voxels):
            x, y, z = indices[0][idx], indices[1][idx], indices[2][idx]
            if (x - central_x)**2 + (y - central_y)**2 + (z - central_z)**2 <= size**2:
                accuracy_map[x, y, z] = np.random.normal(value, std)

    # Create a Nifti image from the accuracy map
    accuracy_map_img = nib.Nifti1Image(accuracy_map, mask_img.affine)

    # Save the accuracy map to disk
    nib.save(accuracy_map_img, output_path)
    print(f"Saved clustered accuracy map to: {output_path}")


def generate_fake_clusters(mask_path: Path, output_directory: Path):
    # Specify parameters for cluster sizes and values
    cluster_sizes = [1, 2, 4]
    cluster_values = [0.9, 0.85, 0.82]  # Accuracy values for each cluster

    # Specify output
    output_path = Path(output_directory, "clustered_accuracy_map.nii.gz")

    # Generate and save the clustered accuracy map
    generate_clustered_accuracy_map(
        mask_path, output_path, cluster_sizes, cluster_values
    )


def generate_random_accuracy_maps(mask_path: Path, output_directory: Path):
    # Example usage:
    mask_path = "brain_mask.nii.gz"
    output_directory = "random_accuracy_maps"
    number_of_maps = 100
    mean_value = 0.5
    std_deviation = 0.01

    # Generate and save random accuracy maps based on the brain mask
    generate_and_save_random_accuracy_maps(
        mask_path, output_directory, number_of_maps, mean_value, std_deviation
    )


if __name__ == "__main__":
    output_directory = Path("fixtures/permutations")
    output_directory.mkdir(parents=True, exist_ok=True)

    # First, let's create a mask
    mask_path = generate_sphere_mask(output_directory)

    # Generate random accuracy maps
    permutation_directory = Path(output_directory, "random_accuracy_maps")
    permutation_directory.mkdir(exist_ok=True)
    generate_and_save_random_accuracy_maps(mask_path, permutation_directory, n_maps=100)

    # Generate an accuracy map that fakes some clusters above chance level
    generate_fake_clusters(mask_path, output_directory)
