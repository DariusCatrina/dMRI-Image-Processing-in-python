from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere, get_fnames
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.viz import actor, window, colormap as cmap

from dipy.segment.clustering import QuickBundles

import numpy as np

class Model(object):
    def __init__(self, DATASET_NAME, use_staford=True, use_dataset=True):
        
        if use_staford:
            self.DATASET_NAME = 'stanford_hardi'
            self.LABLES_NAME = 'stanford_labels'
            f_lables = get_fnames(self.LABLES_NAME)
            self.lables = load_nifti_data(f_lables)
            #white matter
            self.mask = (self.lables == 1) | (self.lables == 2)
        else:
            self.DATASET_NAME = DATASET_NAME
            
        print('Dowloading the data set...')
        f_hardi, f_bvals, f_bvecs = get_fnames(self.DATASET_NAME)
    
        self.data, self.affine = load_nifti(f_hardi)
        
        self.bvals, self.bvecs = read_bvals_bvecs(f_bvals, f_bvecs)
        self.grad_table = gradient_table(self.bvals, self.bvecs)

    def set_mask(self, mask):
        self.mask = mask

    def set_new_affine(self, affine):
        self.affine = affine

    def build_CSA_model(self): # -> peaks, FA_map
        self.csa_model = CsaOdfModel(self.grad_table, sh_order=6)
        self.csa_peaks = peaks_from_model(self.csa_model, self.data, default_sphere,
                                          relative_peak_threshold=.8,
                                          min_separation_angle=45,
                                          mask=self.mask)
        self.FA  = self.csa_peaks.gfa

    def generate_streamlines(self):
        #generating all the streamlines that pass through the white matter
        self.seeds = utils.seeds_from_mask(self.mask, self.affine, density=1)
        self.stopping_criterion = ThresholdStoppingCriterion(self.csa_peaks.gfa, 0.25)
        self.streamline_generator = LocalTracking(self.csa_peaks, self.stopping_criterion, self.seeds,
                                     affine=self.affine, step_size=0.5)

        self.all_streamlines = Streamlines(self.streamline_generator)

    def get_target_streamlines(self, target_mask):
        self.target_mask = target_mask

        self.streamlines = utils.target(self.all_streamlines, self.affine, self.target_mask)
        return Streamlines(self.streamlines)
    
    