"""
3D rendering of the streamlines passing through the calcarine sulcus

Author: Darius Catrina
"""

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

import numpy as np
from scipy.ndimage.morphology import binary_dilation

DATA_SET = 'stanford_hardi'
DATA_LABELS = 'stanford_labels'
DATA_T1 = 'stanford_t1'

print('Dowloading the data set...')
f_hardi, f_bvals, f_bvecs = get_fnames(DATA_SET)
f_labels = get_fnames(DATA_LABELS)
f_t1_names = get_fnames(DATA_T1)

print('Extracting the data set')
data, affine = load_nifti(f_hardi)
labels = load_nifti_data(f_labels)
t1 = load_nifti_data(f_t1_names)
bvals, bvecs = read_bvals_bvecs(f_bvals, f_bvecs)

grad_tab = gradient_table(bvals, bvecs)


#white_matter = binary_dilation((labels == 1) | (labels == 2))
white_matter = (labels == 1) | (labels == 2)

x,y,z = 26, 29, 31
seed_mask_r = np.zeros(data.shape[:-1], 'bool')
seed_mask_l = np.zeros_like(seed_mask_r)
rad = 3
seed_mask_r[x-rad:x+rad, y-rad:y+rad, z-rad:z+rad] = True
x = data.shape[0] - x
seed_mask_l[x-rad:x+rad, y-rad:y+rad, z-rad:z+rad] = True

print('Building the CSA model')
csa_model = CsaOdfModel(grad_tab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=white_matter)

stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

seeds = utils.seeds_from_mask(white_matter, affine, density=1)

streamline_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                     affine=affine, step_size=0.5)

print('Generating the streamlines')
all_streamlines = Streamlines(streamline_generator)


cs_streamlines_r = utils.target(all_streamlines, affine, seed_mask_r)
cs_streamlines_l = utils.target(all_streamlines, affine, seed_mask_l)

streamlines_r = Streamlines(cs_streamlines_r)
streamlines_l = Streamlines(cs_streamlines_l)


streamlines_actor_1 = actor.line(streamlines_r, cmap.line_colors(streamlines_r))
streamlines_actor_2 = actor.line(streamlines_l, cmap.line_colors(streamlines_l))


surface_opacity = 0.5
surface_color = [1, 1, 0]



scene = window.Scene()
scene.add(streamlines_actor_1)
scene.add(streamlines_actor_2)


vol_actor1 = actor.slicer(t1, affine=affine)

vol_actor1.display(x=data.shape[0] // 2)
vol_actor2 = vol_actor1.copy()
vol_actor2.display(z=data.shape[2] // 2)
scene.add(vol_actor1)
scene.add(vol_actor2)


window.show(scene)

window.record(scene, out_path='streamlines_from_calcarine_sulcus.png', size=(1200, 900))