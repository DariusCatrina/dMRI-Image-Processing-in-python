# Data retrieval/loading from:
#       [x] files
#       [] drive 
# [] FA extraction
# Data augemntation:
#       [] X,Y,Z rotation
#       [] Minimum size reduction
#       [] ± Segmentation per Bundle
# Extra features extraction:
#       [] Symmetry percentage / Shape Similarity
#       [] Left and Right length 
# Data Parsing in Training, Validation, Testing sets

import os
import numpy as np

from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from dipy.core.gradients import gradient_table

from dipy.reconst.shm import CsaOdfModel


class NiftiProccesing(object):
    def __init__(self, dir_name, subject_list):
        self.data_file = os.path.abspath(os.path.join("",f'{dir_name}/'))
        self.subject_list = subject_list       
    
    def load_subjects(self, _type, b=True):
        print('Loading the subjects...')
        # downsampled img shape: (128, 210, 128, 49), affine shape: (4, 4)
        imgs = np.zeros((len(self.subject_list), 128,210,128,49))
        affines = np.zeros((len(self.subject_list), 4, 4))
        if b:
            grad_tables = np.empty(len(self.subject_list), dtype=object)


        for i, subject in enumerate(self.subject_list):
            print(f'\tUploading the {subject} subject')
            subject_fname = os.path.join(self.data_file, subject+_type)
            
            scans, affine, vox_size = load_nifti(subject_fname, return_voxsize=True)
            new_vox_size = [dim * 2 for dim in vox_size]
            img, affine = reslice(scans, affine, vox_size, new_vox_size, order=1)
            
            imgs[i] = img
            affines[i] = affine
            if b:
                bvals_fname = os.path.join(self.data_file, subject+'_bvals_fix.txt')
                bvecs_fname = os.path.join(self.data_file, subject+'_bvec_fix.txt')

                bvals, bvecs = np.loadtxt(bvals_fname), np.loadtxt(bvecs_fname)

                bvec_orient = [-2,1,3]
                bvec_sign = bvec_orient/np.abs(bvec_orient)
                bvecs = np.c_[bvec_sign[0]*bvecs[:, np.abs(bvec_orient[0])-1], bvec_sign[1]*bvecs[:, np.abs(bvec_orient[1])-1],bvec_sign[2]*bvecs[:, np.abs(bvec_orient[2])-1]]

                grad_tab = gradient_table(bvals, bvecs)
                grad_tables[i] = grad_tab
        
        return imgs, affines, grad_tables
    
    def fa_extraction(self, grad_table):
        csa_model = CsaOdfModel(grad_table, sh_order=6)



class Dataset(NiftiProccesing):
    def __init__(self, dir_name='TEST_SUBJECT', subject_list = ['N57709']):
        NiftiProccesing.__init__(self, dir_name, subject_list)
        self.types = {'scan' : '_nii4D_RAS.nii.gz', 
                      'binary_mask' : '_dwi_binary_mask.nii.gz', 
                      'lables' : '_chass_symmetric3_labels_RAS_lr_ordered.nii.gz'}

        self.imgs, self.affines, self.grad_tables = self.load_subjects(_type=self.types['scan'], b=True)





dataset = Dataset()