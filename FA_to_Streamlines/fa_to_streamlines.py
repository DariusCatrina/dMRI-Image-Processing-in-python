"""
3D rendering of the FA values mapped on the streamlines of the calcarine sulcus ROI

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
from dipy.tracking.streamline import set_number_of_points

import numpy as np
import matplotlib.pyplot as plt


DATA_SET = 'stanford_hardi'
DATA_LABELS = 'stanford_labels'
DATA_T1 = 'stanford_t1'

print('Dowloading the data set...')
f_hardi, f_bvals, f_bvecs = get_fnames(DATA_SET)
f_labels = get_fnames(DATA_LABELS)
f_t1_names = get_fnames(DATA_T1)

print('Extracting the data set')
data, _ = load_nifti(f_hardi)
labels = load_nifti_data(f_labels)
t1 = load_nifti_data(f_t1_names)
bvals, bvecs = read_bvals_bvecs(f_bvals, f_bvecs)

grad_tab = gradient_table(bvals, bvecs)

affine = np.eye(4)

white_matter = (labels == 1) | (labels == 2)

#Masking the left and right side of the calcarine sulcus ROI
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
FA_map = csa_peaks.gfa # (x,y,z) x number of pixels

seeds = utils.seeds_from_mask(white_matter, affine, density=1)

streamline_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                     affine=affine, step_size=0.5)

print('Generating the streamlines')
all_streamlines = Streamlines(streamline_generator)

cs_streamlines_r = utils.target(all_streamlines, affine, seed_mask_r)
cs_streamlines_l = utils.target(all_streamlines, affine, seed_mask_l)

streamlines_r = set_number_of_points(Streamlines(cs_streamlines_r), 50)
streamlines_l = set_number_of_points(Streamlines(cs_streamlines_l), 50)

#mapping function
def map_streamlines_to_fa(fa_map, streamlines):
    fa_streamlines = []
    for i, streamline in enumerate(streamlines):
        #fa points per streamline
        fa_streamline = [fa_map[int(p[0]),int(p[1]),int(p[2])] for p in streamline] #[[x,y,z],[],[]]],[[]]
        #mean Fa per streamline
        fa_streamline = np.mean(fa_streamline)
        fa_streamlines.append(fa_streamline)

    return np.array(fa_streamlines)



print('Mapping FA to streamlines')
FA_streamlines_r = map_streamlines_to_fa(FA_map, streamlines_r)
FA_streamlines_l = map_streamlines_to_fa(FA_map, streamlines_l)


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_str():
    #fa_r,fa_l = [], []
    points = np.linspace(0, len(streamlines_r), num=50)
    for i, streamline in enumerate(streamlines_r):
        fa_points = [FA_map[int(p[0]),int(p[1]),int(p[2])] for p in streamline]
        plt.plot(points, fa_points,  color='#73b504')
        if i == 5:
            break
        #fa_r.append(fa_points)
    for i, streamline in enumerate(streamlines_l):
        fa_points = [FA_map[int(p[0]),int(p[1]),int(p[2])] for p in streamline]
        plt.plot(points, fa_points, color='#800080')
        if i == 5:
            break
        #fa_l.append(fa_points)
    legend_elements = [Line2D([0], [0], color='#73b504', lw=4, label='Right streamline'),
                       Line2D([0], [0], color='#800080', lw=4, label='Left streamline')]
    plt.ylabel('FA')
    plt.xlabel('Points on streamline')
    plt.title('FA along streamlines projection(5 samples)')
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()

def plt_hitogram():
    fa_streamlines = []
    for i, streamline in enumerate(streamlines_r):
        #fa points per streamline
        fa_streamline = [FA_map[int(p[0]),int(p[1]),int(p[2])] for p in streamline] #[[x,y,z],[],[]]],[[]]
        #mean Fa per streamline
        fa_streamline = np.mean(fa_streamline)
        fa_streamlines.append(fa_streamline)
        if i == 100:
            break
    fa_streamlines_l = []
    for i, streamline in enumerate(streamlines_l):
        #fa points per streamline
        fa_streamline = [FA_map[int(p[0]),int(p[1]),int(p[2])] for p in streamline] #[[x,y,z],[],[]]],[[]]
        #mean Fa per streamline
        fa_streamline = np.mean(fa_streamline)
        fa_streamlines_l.append(fa_streamline)
        if i == 100:
            break

    plt.bar(np.arange(len(fa_streamlines_l)),fa_streamlines_l, align='edge', width=1.0, color='#800080')
    plt.bar(np.arange(len(fa_streamlines)),fa_streamlines, align='edge', width=1.0, color='#73b504')

    plt.show()


    
plot_str()
plt_hitogram()

def viz(interactive):
    if interactive == True:
        streamlines_actor_1 = actor.line(streamlines_r, FA_streamlines_r)#cmap.line_colors(streamlines_r))
        streamlines_actor_2 = actor.line(streamlines_l,FA_streamlines_l)#cmap.line_colors(streamlines_l))

        surface_opacity = 0.5
        surface_color = [1, 1, 0]



        scene = window.Scene()
        scene.add(streamlines_actor_1)
        scene.add(streamlines_actor_2)


        vol_actor1 = actor.slicer(t1, affine=affine)

        vol_actor1.display(x=data.shape[0] // 2)
        bar = actor.scalar_bar()
        scene.add(bar)
        scene.add(actor.axes(scale=(30,30,30)))

        window.show(scene)
        

viz(False)

