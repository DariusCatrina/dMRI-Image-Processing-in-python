# Data(scans, b-vects,  b-values, b0_masks) retrieval/loading from:
#       [x] files (from path/directory provided)
#       [] drive 
# [x] FA extraction (saved to file=BACKUP_DATA/GFA_IMAGES.npy)
# [x] Connectome generation (saved to file=BACKUP_DATA/STREAMLINES.npy)
# Data augemntation:
#       [x] X,Y,Z rotation (saved to file=BACKUP_DATA/GFA_X/Y/Z_ROT_IMAGES.npy)
#       [] Minimum size reduction / Batch normalization
#       [] Creat single array with the whole data
#       [] ± Segmentation per Bundle (divide all the streamlines in bundles for multiple inputs for CNN)
# Streamlines:
#       [x] Streamline extraction (saved to file=BACKUP_DATA/STREAMLINES.npy)
#       [x] Set constant number of points
# Extra features extraction:
#       [] Symmetry percentage / Shape Similarity (TODO)
#       [] Left and Right length (TODO)
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
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import set_number_of_points


class NiftiProccesing(object):
    def __init__(self, dir_name, subject_list, global_path):
        if global_path == True:
            self.data_directory = dir_name
        else:
            self.data_directory = os.path.abspath(os.path.join("",f'{dir_name}/'))
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
            subject_fname = os.path.join(self.data_directory, subject+_types['scan'])
            binary_mask_fname = os.path.join(self.data_directory, subject+_types['binary_mask'])

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


            bvals_fname = os.path.join(self.data_directory, subject+'_bvals_fix.txt')
            bvecs_fname = os.path.join(self.data_directory, subject+'_bvec_fix.txt')

            bvals, bvecs = np.loadtxt(bvals_fname), np.loadtxt(bvecs_fname)

            bvec_orient = [-2,1,3]
            bvec_sign = bvec_orient/np.abs(bvec_orient)
            bvecs = np.c_[bvec_sign[0]*bvecs[:, np.abs(bvec_orient[0])-1], bvec_sign[1]*bvecs[:, np.abs(bvec_orient[1])-1],bvec_sign[2]*bvecs[:, np.abs(bvec_orient[2])-1]]

            grad_tab = gradient_table(bvals, bvecs)
            grad_tables[i] = grad_tab
        
        return imgs, affines, masks, grad_tables
    
    def fa_extraction(self, imgs, grad_table, mask, subject):
        #Extracts FA map for a subject
        print(f'\tExtracting FA for {subject}')
        csa_model = CsaOdfModel(grad_table, sh_order=6)
        csa_peaks = peaks_from_model(csa_model, imgs, default_sphere,
                             relative_peak_threshold=.25,
                             min_separation_angle=25,
                             mask=mask)

        return csa_peaks
    
    def streamlines_extraction(self, fa, b0_masks, affine, csa_peaks, subject,no_points,ratio=None):
        print(f'\tExtracting streamlines for {subject}')
        stopping_crit = ThresholdStoppingCriterion(fa, 0.25)
        seeds = utils.seeds_from_mask(b0_masks, affine, density=1)

        streamline_generator = LocalTracking(csa_peaks, stopping_crit, seeds,
                                     affine=affine, step_size=0.1)

        return set_number_of_points(Streamlines(streamline_generator), no_points)
    
    def rotate_img(self, img, angle, axes, subject):
        #Rotate a 3D image volume for data augmentation...
        from scipy.ndimage.interpolation import rotate
        
        return rotate(input=img, angle=angle, axes=axes)

    def fa_mapping(self, streamlines, fa, no_of_points):
        fa_points = np.zeros((len(streamlines), no_of_points), dtype=np.int64)
        for i, streamline in enumerate(streamlines):
            for ii, (x,y,z) in enumerate(streamline):
                fa_points[i][ii] = fa[int(x),int(y),int(z)]

        return fa_points



class Dataset(NiftiProccesing):
    def __init__(self, augmentation_factor=5 , dir_name='TEST_SUBJECT',global_path=False ,subject_list = ['N57709'], load_subjects=True):
        NiftiProccesing.__init__(self, dir_name, subject_list, global_path)
        self.types = {'scan' : '_nii4D_RAS.nii.gz', 
                      'binary_mask' : '_dwi_binary_mask.nii.gz', 
                      'lables' : '_chass_symmetric3_labels_RAS_lr_ordered.nii.gz'}
        self.subject_list = subject_list
        if load_subjects:
            self.imgs, self.affines, self.masks, self.grad_tables = self.load_subjects(_types=self.types)
        self.gfa_imgs = np.zeros((len(subject_list), 128,210,128))
        self.csa_peaks = np.zeros(len(subject_list), dtype=object)
        self.streamlines = np.zeros(len(subject_list), dtype=object)
        self.fa_streamlines = np.zeros(len(subject_list), dtype=object)
        self.augmentation_factor = augmentation_factor
        self.rot_angle_pass = 360/self.augmentation_factor
        # Each rotation the image will have different shapes
        self.x_rot_imgs, self.y_rot_imgs, self.z_rot_imgs = np.zeros(self.augmentation_factor*len(subject_list), dtype=object),  np.zeros(self.augmentation_factor*len(subject_list), dtype=object),  np.zeros(self.augmentation_factor*len(subject_list), dtype=object)

    def populate_fa(self, save_data):
        print('FA maps generation...')
        for i in range(len(self.subject_list)):
            self.csa_peaks[i] = self.fa_extraction(self.imgs[i], self.grad_tables[i], self.masks[i], self.subject_list[i])
            self.gfa_imgs[i] = self.csa_peaks[i].gfa
            if save_data:
                np.save('BACKUP_DATA/GFA_IMAGES', self.gfa_imgs)

    def populate_streamlines(self, save_data=True):
        print('Streamline generation...')
        for i in range(len(self.subject_list)):
            self.streamlines[i] = self.streamlines_extraction(self.gfa_imgs[i],self.masks[i], self.affines[i],self.csa_peaks[i],self.subject_list[i], no_points=100)
            self.fa_streamlines[i] = self.fa_mapping(self.streamlines[i], self.gfa_imgs[i], no_of_points=100)
            if save_data:
                np.save('BACKUP_DATA/STREAMLINES', self.streamlines)

    def apply_augemntation(self, save_data=True):
        #save_data is for testing only: it saves data after the augmentation has been done in case something fails
        #rotation axes 
        x_rot = (1,0)
        z_rot = (2,0)
        y_rot = (2,1)

        print('Data augmentation...')
        for i in range(len(self.subject_list)):
            # 4 X rot, 4 Y rot, 4 Z rot
            print(f'\tRotation of the {self.subject_list[i]} subject')
            for j in range(0, self.augmentation_factor - 1):
                self.x_rot_imgs[i+j] = self.rotate_img(self.gfa_imgs[i], (j + 1) * self.rot_angle_pass, x_rot, self.subject_list[i])
                self.y_rot_imgs[i+j] = self.rotate_img(self.gfa_imgs[i], (j + 1) * self.rot_angle_pass, y_rot, self.subject_list[i])
                self.z_rot_imgs[i+j] = self.rotate_img(self.gfa_imgs[i], (j + 1) * self.rot_angle_pass, z_rot, self.subject_list[i])
            if save_data:                
                np.save('BACKUP_DATA/GFA_X_ROT_IMAGES', self.x_rot_imgs)
                np.save('BACKUP_DATA/GFA_Y_ROT_IMAGES', self.y_rot_imgs)
                np.save('BACKUP_DATA/GFA_Z_ROT_IMAGES', self.z_rot_imgs)

    def upload_imgs(self):
        print('Loading the images')
        self.gfa_imgs = np.load('BACKUP_DATA/GFA_IMAGES.npy')
        self.x_rot_imgs = np.load('BACKUP_DATA/GFA_X_ROT_IMAGES.npy',allow_pickle=True)
        self.y_rot_imgs = np.load('BACKUP_DATA/GFA_Y_ROT_IMAGES.npy',allow_pickle=True)
        self.z_rot_imgs = np.load('BACKUP_DATA/GFA_Z_ROT_IMAGES.npy',allow_pickle=True)

    def upload_streamlines(self):
        print('Loading the streamlines')
        self.streamlines = np.load('BACKUP_DATA/STREAMLINES',allow_pickle=True)

    def display(self, index):
        mid_x = self.gfa_imgs.shape[-1] // 2
        mid_y = self.gfa_imgs.shape[-3] // 2
        mid_z = self.gfa_imgs.shape[-2] // 2


        fig, axs = plt.subplots(2,3)
        axs[0,0].imshow(self.gfa_imgs[index,:,:,mid_x].T,cmap='gray')
        axs[0,0].title.set_text('From x perspective')
        axs[0,1].imshow(self.gfa_imgs[index,:,mid_z,:],cmap='gray')
        axs[0,1].title.set_text('From z perspective')
        axs[0,2].imshow(self.gfa_imgs[index,mid_y,:,:],cmap='gray')
        axs[0,2].title.set_text('From y perspective')

        axs[1,0].imshow(self.x_rot_imgs[index + 2][:,:,mid_x].T,cmap='gray')
        axs[1,1].imshow(self.z_rot_imgs[index + 2][:,mid_z,:],cmap='gray')
        axs[1,2].imshow(self.y_rot_imgs[index + 2][mid_y,:,:],cmap='gray')


        plt.show()



def genotype_extraction(age_limit, plot=False):
    #! wget https://raw.githubusercontent.com/portokalh/skullstrip/master/book_keeping/QCLAB_AD_mice062921.csv
    df = pd.DataFrame(pd.read_csv('QCLAB_AD_mice062921.csv'))
    #df = df.dropna(axis="columns", how="any")

    limit = df.shape[0] - 4
    data_dic = []

    for idx, row in df[['DWI', 'Genotype', 'Age_Months']].iterrows():
        if idx >= limit:
            break
        name = row['DWI']
        gene = row['Genotype']
        
        if name not in ['Blank','Died', 'NaN'] and gene in ['APOE33', 'APOE22', 'APOE44']:
            age =  float(row['Age_Months'])
            if age <= age_limit:
                dict_ = {'subject':name, 'genotype': gene, 'age':age}
                data_dic.append(dict_)

    data = pd.DataFrame(data_dic)
    data.to_csv('DATA.csv', index=False)

    #pie chart: gene distribution
    if plot:
        from matplotlib.pyplot import pie, axis, show
        def f(x):
            return len(list(x))
        sums = data.groupby(data["genotype"])['subject'].apply(f)
        print(sums)
        axis('equal')
        pie(sums, labels=sums.index);
        show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--GLOBAL_PATH', type=bool, default=False, help='If the directory is local or a global path') 
    parser.add_argument('--SUBJ_DIR', type=str, default='TEST_SUBJECT', help='Directory/Folder location of the subjects') 
    parser.add_argument('--AUGM_FACT', type=int, default=10, help='Augmentation Factor')
    parser.add_argument('--AGE_LIMIT', type=int, default=21, help='Age limit of the subjects')
    args = parser.parse_args()

    #Subject list extraction
    # genotype_extraction(age_limit=args.AGE_LIMIT, plot=False)
    # df  = pd.DataFrame(pd.read_csv('DATA.csv'))
    # subject_list = df[df.columns[0]].tolist()
    subject_list = ['N57709']

    #Data
    dataset = Dataset(load_subjects=True,
                      global_path=args.GLOBAL_PATH,
                      augmentation_factor=args.AUGM_FACT, 
                      dir_name=args.SUBJ_DIR,
                      subject_list=subject_list)

    dataset.populate_fa(save_data=True)
    dataset.populate_streamlines(save_data=True)
    dataset.apply_augemntation()
    dataset.display(0)