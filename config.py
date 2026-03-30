import os

class Config:
    # Data root directory — set this to your data location
    root_dir = os.environ.get('DATA_ROOT', './data')

    # CHA dataset: MRI data directories
    # Each directory should contain .nii.gz files named by subject ID
    CHA_data_dir = {
        'T1w': os.path.join(root_dir, 'CHA/sMRI_brain/'),
        'FA': os.path.join(root_dir, 'CHA/FA/'),
        'MD': os.path.join(root_dir, 'CHA/MD/'),
        'RD': os.path.join(root_dir, 'CHA/RD/'),
        'AD': os.path.join(root_dir, 'CHA/AD/'),
    }

    # CHA dataset: phenotype/metadata CSV files
    CHA_phenotype_dir = {
        'ASDvsGDD': os.path.join(root_dir, 'CHA/metadata/ASDGDD_QC_quotient.csv'),
    }

    # Stratification labels for iterative stratification
    iter_strat_label = {
        "binary": ['sex', 'Autism'],
        "multiclass": ['Mental_quotient', 'Motor_quotient'],
        "continuous": ["age(days)"]
    }

    # PSM covariates
    psm_label = ['age(days)', 'Mental_quotient', 'Motor_quotient']

    # Confounders for distance correlation monitoring
    confounders = ['sex', 'age(days)', 'Mental_quotient', 'Motor_quotient']

    # Dataset registry
    data_dict = {'CHA': CHA_data_dir}
    phenotype_dict = {'CHA': CHA_phenotype_dir}
