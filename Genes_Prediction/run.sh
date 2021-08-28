git clone https://github.com/DariusCatrina/dMRI-Image-Processing-in-python.git
cd dMRI-Image-Processing-in-python/Genes_Prediction
mkdir BACKUP_DATA
pip install -r requirements.txt
# Some notations:
# 
# subj_dir_name : The directory/Path where all the subjects are saved(ex: APOE_TRK)
# global_path : bool, only True/False, if the subj_dir_name is a global path 
#               or a directory saved in Gene_Prediciton file
# age_limit : age limit of the subjects used for training
# agm_fct : augmentation factor(ex.: 5), i.e. how many rotations per axe it will generate for an image

# All of these variables(subj_dir_name, global_path, age_limit, agm_fct) have to be substituted with 
#              real ones, following the command below

python data_processing.py --GLOBAL_PATH global_path  --SUBJ_DIR subj_dir_name --AGE_LIMIT age_limit --AUGM_FACT agm_fct

