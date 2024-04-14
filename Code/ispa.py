import os
import re
import glob
import numpy as np
import nibabel as nb
from nilearn.image import concat_imgs, clean_img
from nilearn.decoding import SearchLight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

class ISPA:
    """
    Inter-Subject Pattern Analysis (ISPA) class for fMRI analysis.
    """

    def __init__(self, data_dir, subjects_to_include=None):
        """
        Initialize the ISPA object.

        Parameters:
        - data_dir (str): Path to the directory containing GLM data.
        - subjects_to_include (list or None): List of subject identifiers to include. 
          If None, all subjects found in `data_dir` will be included.
        """
        self.data_dir = data_dir
        self.subjects_to_include = subjects_to_include if subjects_to_include else []
        self.fmri_nii_dict = {}
        self.concat_imgs_stand_dict = {}

    def _standardize_images(self):
        """
        Concatenate and standardize beta images per subject and run.
        """
        for subj, subj_value in self.fmri_nii_dict.items():
            for run, run_value in subj_value.items():
                # Collect all beta images for the current subject and run
                images_to_concat = [item['image'] for item in run_value]
                
                # Concatenate beta images along the time axis (4th dimension)
                concatenated_image = concat_imgs(images_to_concat)
                
                # Clean and standardize the concatenated image
                stand_image = clean_img(concatenated_image, standardize=True, detrend=False)
                
                # Store the standardized image in the dictionary
                self.concat_imgs_stand_dict.setdefault(subj, {})[run] = stand_image

    def load_data(self):
        """
        Load beta map data from GLM directory.
        """
        files = os.listdir(self.data_dir)
        pattern = r'sub-(\d+)_LSA'
        subject_numbers = [re.search(pattern, file_name).group(1) for file_name in files if re.search(pattern, file_name)]

        beta_flist = []
        y = []
        subj_vect = []
        runs = []

        for subject_number in subject_numbers:
            if not self.subjects_to_include or subject_number in self.subjects_to_include:
                subject_directory = os.path.join(self.data_dir, f'sub-{subject_number}_LSA')

                beta_map_files = glob.glob(os.path.join(subject_directory, f'sub-{subject_number}_run*_*.nii.gz'))
                beta_map_files.sort()
                beta_flist.extend(beta_map_files)

                for beta_map_file in beta_map_files:
                    label = beta_map_file.split('_')[-3]
                    if "InfOther" in label:
                        label = "Other"
                    elif "InfOwn" in label:
                        label = "Own"
                    y.append(label)
                    subj_vect.append(subject_number)

                for beta_map_file in beta_map_files:
                    run = beta_map_file.split('_')[-4]
                    runs.append(run)

                for beta_path in beta_flist:
                    parts = re.split(r'[/_]', beta_path)
                    subj = parts[-6]
                    run = 'run-' + parts[-4]
                    condition = parts[-3]
                    beta = parts[-2]

                    image = nb.load(beta_path)

                    if subj in self.fmri_nii_dict:
                        if run in self.fmri_nii_dict[subj]:
                            self.fmri_nii_dict[subj][run].append({'condition': condition, 'beta': beta, 'image': image})
                        else:
                            self.fmri_nii_dict[subj][run] = [{'condition': condition, 'beta': beta, 'image': image}]
                    else:
                        self.fmri_nii_dict[subj] = {run: [{'condition': condition, 'beta': beta, 'image': image}]}

    def run_searchlight(self, output_dir, searchlight_radius=4, n_jobs=-1):
        """
        Run searchlight decoding.

        Parameters:
        - output_dir (str): Output directory to save results.
        - searchlight_radius (int): Radius of the searchlight sphere (in voxels).
        - n_jobs (int): Number of parallel jobs for searchlight processing.
        """
        loso = LeaveOneGroupOut()
        unique_subjects = np.unique([subj for subj in self.fmri_nii_dict.keys()])
        n_splits = loso.get_n_splits(groups=unique_subjects)
        chance_level = 1. / len(np.unique(y))

        for split_ind, (train_inds, test_inds) in enumerate(loso.split(subj_vect, subj_vect, subj_vect)):
            y_train = np.array(y)[train_inds]

            clf = LogisticRegression()
            searchlight = SearchLight(
                process_mask_img=_get_process_mask_image(),  # Provide process mask image
                radius=searchlight_radius,
                n_jobs=n_jobs,
                verbose=1,
                cv=[(train_inds, test_inds)],
                estimator=clf
            )

            searchlight.fit(_get_all_betas_stand(), y_train)

            score_map = searchlight.scores_ - chance_level
            output_filename = f'ispa_searchlight_accuracy_split{split_ind+1:02d}_of_{n_splits:02d}.nii'
            output_path = os.path.join(output_dir, output_filename)
            nb.save(nb.Nifti1Image(score_map, _get_resampled_mask_brain()), output_path)

    def _get_all_betas_stand(self):
        """
        Concatenate all standardized beta images.
        """
        all_images_to_concat = []

        for subj, subj_value in self.concat_imgs_stand_dict.items():
            for run, stand_image in subj_value.items():
                all_images_to_concat.append(stand_image)

        return concat_imgs(all_images_to_concat)

    def _get_process_mask_image(self):
        """
        Return the process mask image (resampled if necessary).
        """
        mask_brain_path = "/path/to/brain_mask.nii.gz"
        mask_brain = nb.load(mask_brain_path)
        beta_img = nb.load(next(iter(self.fmri_nii_dict.values()))[next(iter(self.fmri_nii_dict.values()))][0]['image_path'])
        if not beta_img.shape == mask_brain.shape:
            resampled_mask_brain = nb.resample_to_img(mask_brain, beta_img)
            return resampled_mask_brain
        else:
            return mask_brain

    def _get_resampled_mask_brain(self):
        """
        Resample ROI mask image to match beta images.
        """
        mask_brain_path = "/path/to/brain_mask.nii.gz"
        mask_brain = nb.load(mask_brain_path)
        beta_img = nb.load(next(iter(self.fmri_nii_dict.values()))[next(iter(self.fmri_nii_dict.values()))][0]['image_path'])
        resampled_mask_brain = nb.resample_to_img(mask_brain, beta_img)
        return resampled_mask_brain



if __name__ == "__main__":
    data_dir = '/path/to/data_directory'
    mask_path = '/path/to/brain_mask.nii.gz'

    ispa = ISPAAnalysis(data_dir)
    ispa.run_analysis(mask_path)
