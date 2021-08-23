# Data retrieval/loading from:
#       [x] files
#       [] drive 
# [x] FA extraction
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
from matplotlib import pyplot as plt
import pandas as pd

from dipy.io.image import load_nifti
from dipy.align.reslice import reslice
from dipy.core.gradients import gradient_table

from dipy.reconst.shm import CsaOdfModel
from dipy.segment.mask import median_otsu
from dipy.direction import peaks_from_model
from dipy.data import default_sphere



class NiftiProccesing(object):
    def __init__(self, dir_name, subject_list):
        self.data_file = os.path.abspath(os.path.join("",f'{dir_name}/'))
        self.subject_list = subject_list       
    
    def load_subjects(self, _types):
        print('Loading the subjects...')
        # downsampled img shape: (128, 210, 128, 49), affine shape: (4, 4)
        imgs = np.zeros((len(self.subject_list), 128,210,128,49))
        affines = np.zeros((len(self.subject_list), 4, 4))
        masks = np.zeros((len(self.subject_list), 128,210,128))
        grad_tables = np.empty(len(self.subject_list), dtype=object)

        #reading the scans, binary masks, bvalues and bvectors for every subject
        for i, subject in enumerate(self.subject_list):
            print(f'\tUploading the {subject} subject')
            subject_fname = os.path.join(self.data_file, subject+_types['scan'])
            binary_mask_fname = os.path.join(self.data_file, subject+_types['binary_mask'])

            scans, affine, vox_size = load_nifti(subject_fname, return_voxsize=True)
            b0_mask, _ , vox_b0= load_nifti(binary_mask_fname, return_voxsize=True)
            new_vox_size = [dim * 2 for dim in vox_size]
            new_vox_b0 = [dim * 2 for dim in vox_b0]
            #downsampling the binary mask and the dMRI scans
            img, affine = reslice(scans, affine, vox_size, new_vox_size, order=1)
            b0_mask, _ = reslice(b0_mask, affine, vox_b0, new_vox_b0, order=1)

            imgs[i] = img
            affines[i] = affine
            masks[i] = b0_mask


            bvals_fname = os.path.join(self.data_file, subject+'_bvals_fix.txt')
            bvecs_fname = os.path.join(self.data_file, subject+'_bvec_fix.txt')

            bvals, bvecs = np.loadtxt(bvals_fname), np.loadtxt(bvecs_fname)

            bvec_orient = [-2,1,3]
            bvec_sign = bvec_orient/np.abs(bvec_orient)
            bvecs = np.c_[bvec_sign[0]*bvecs[:, np.abs(bvec_orient[0])-1], bvec_sign[1]*bvecs[:, np.abs(bvec_orient[1])-1],bvec_sign[2]*bvecs[:, np.abs(bvec_orient[2])-1]]

            grad_tab = gradient_table(bvals, bvecs)
            grad_tables[i] = grad_tab
        
        return imgs, affines, masks ,grad_tables
    
    def fa_extraction(self, imgs, grad_table, mask, subject):
        print(f'Extracting FA for {subject}')
        csa_model = CsaOdfModel(grad_table, sh_order=6)
        csa_peaks = peaks_from_model(csa_model, imgs, default_sphere,
                             relative_peak_threshold=.25,
                             min_separation_angle=25,
                             mask=mask)

        return csa_peaks.gfa
    
    def rotate_img(self, img, angles):
        #Rotate a 3D image volume for data augmentation...
        pass


class Dataset(NiftiProccesing):
    def __init__(self, dir_name='TEST_SUBJECT', subject_list = ['N57709']):
        NiftiProccesing.__init__(self, dir_name, subject_list)
        self.types = {'scan' : '_nii4D_RAS.nii.gz', 
                      'binary_mask' : '_dwi_binary_mask.nii.gz', 
                      'lables' : '_chass_symmetric3_labels_RAS_lr_ordered.nii.gz'}

        imgs, affines, masks, grad_tables = self.load_subjects(_types=self.types)

        self.gfa_imgs = np.zeros((len(subject_list), 128,210,128))

        for i in range(len(subject_list)):
            self.gfa_imgs[i] = self.fa_extraction(imgs[i], grad_tables[i], masks[i], subject_list[i])

    def display(self, index):
        _slice = self.gfa_imgs.shape[-1] // 2
        plt.imshow(self.gfa_imgs[index,:,:,_slice].T,cmap='gray')
        plt.show()





dataset = Dataset()
dataset.display(0)

def genotype_extraction():
    #! wget https://raw.githubusercontent.com/portokalh/skullstrip/master/book_keeping/QCLAB_AD_mice062921.csv
    df = pd.DataFrame(pd.read_csv('QCLAB_AD_mice062921.csv'))
    #df = df.dropna(axis="columns", how="any")

    limit = df.shape[0] - 4
    data_dic = []

    for idx, row in df[['DWI', 'Genotype']].iterrows():
        if idx >= limit:
            break
        name = row['DWI']
        gene = row['Genotype']
        if name not in ['Blank','Died', 'NaN'] and gene in ['APOE33', 'APOE22', 'APOE44']:
            dict_ = {'subject':name, 'genotype': gene}
            data_dic.append(dict_)

    data = pd.DataFrame(data_dic)
    data.to_csv('DATA.csv', index=False)

    #pie chart: gene distribution
    from matplotlib.pyplot import pie, axis, show
    def f(x):
        return len(list(x))
    sums = data.groupby(data["genotype"])['subject'].apply(f)
    print(sums)
    axis('equal')
    pie(sums, labels=sums.index);
    show()


genotype_extraction()