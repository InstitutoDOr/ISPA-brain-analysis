ISPA - SearchLight
Script adapted from: https://github.com/SylvainTakerkart/inter_subject_pattern_analysis/tree/master/fmri_data

Original paper: https://doi.org/10.1016/j.neuroimage.2019.116205

The script (Code/ispa.py) demonstrates how to use the searchlight implementation available in nilearn to perform group-level decoding using an inter-subject pattern analysis (ISPA) scheme.


For statistical analysis we have implemented the permutatoin scheme suggested by Stelzer et al. 2013, by permuting labels for X searchlights per subject (Code/permutation_stats.py).
