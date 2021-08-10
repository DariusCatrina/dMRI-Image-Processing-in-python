'''
Script that returns symmetry related features like: shape similarity score, length comparison, and plots like: FA along a 
left and right streamlines, histogram of the FA, length histogram etc.

The script runs by default on HCP842 Atlas, but can be change when running it in CLI.

Author: Darius Catrina
'''
import os

from dipy.io.streamline import load_trk         
from dipy.viz import actor, window, colormap   
from dipy.io.image import load_nifti            
from dipy.tracking.streamline import transform_streamlines, set_number_of_points
from dipy.tracking.utils import length   
from dipy.segment.bundles import bundle_shape_similarity

import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.streamline import save_tractogram
from dipy.align.reslice import reslice


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def load_nifti_format(file_name, directory, downsample=False) -> tuple:
    """
    Parameters
    --------
    file_name: string, Name of the file of the NIFTI file read
    directory: string, Name of the drectory where the image is saved
    downsample: bool

    Returns
    --------
    data: Data stored in the NIFTI file as an numpy array
    affine: Affine of the data
    """
    data_file = os.path.abspath(os.path.join("",f'{directory}/')) #Directory where the nifti file is stored
    bundle_file_name = os.path.join(data_file, file_name)

    if downsample:
        imgs, affine0, vox_size = load_nifti(bundle_file_name, return_voxsize=True)
        new_vox_size = [dim *2 for dim in vox_size]
        data, affine = reslice(imgs, affine0, vox_size, new_vox_size, order=1)
        return [data, affine]
    else:
        return  load_nifti(bundle_file_name)

def load_bundle(bundle_name, fa_affine, transfrom=True):
    """
    Parameters
    --------
    bundle_name: Name of the file of the bundle, the file has to be ".trk" type
    fa_affine: The bundles will be translated acording to this affine matrix
    transfrom: bool value that represents whether the streamlines will pe transposed or not

    Returns
    --------
    new_streamlines: The transformed stremlines of the bundle 

    """
    data_file = os.path.abspath(os.path.join("",'DATA/Atlas_80_Bundles/bundles'))
    bundle_file_name = os.path.join(data_file, bundle_name)
    bundle = load_trk(bundle_file_name, 'same', bbox_valid_check=False)
    if transfrom:
        new_streamlines = transform_streamlines(bundle.streamlines, np.linalg.inv(fa_affine))
        new_streamlines = set_number_of_points(new_streamlines, 100)
    else:
        new_streamlines=bundle.streamlines
    return new_streamlines

def get_fa_along_streamlines(bundle_l, bundle_r, fa):
    """
    Registers and maps the FA values along left and right bundles

    Parameters
    --------
    bundle_l: The left bundle. Data type: 2D numpy array, shape: (N,3) where N is the no of streamlines
    bundle_r: The right bundle. Data type: 2D numpy array, shape: (N,3) where N is the no of streamlines
    fa: FA matrix of the entire brain

    Returns
    --------
    fa_streamlines_r: Fa values mapped along a streamline(for every point along the streamline)
    fa_streamlines_l: Fa values mapped along a streamline(for every point along the streamline)
    """

    fa_streamlines_r = np.empty((len(bundle_r), 100)) #[fa1,fa2,...],[],[] -> no of streamlines
    fa_streamlines_l = np.empty((len(bundle_l), 100))

    import itertools
    for i, (streamline_l, streamline_r) in enumerate(itertools.zip_longest(bundle_l, bundle_r)):
        if streamline_l is not None:
            #[[x1,y1,z1],[x2,y2,z2]] -> [fa1, fa2]
            for ii,(x,y,z) in enumerate(streamline_l):
                fa_streamlines_l[i][ii] = fa[int(x),int(y),int(z)] #fa of every point 

        if streamline_r is not None:
            for jj,(x,y,z) in enumerate(streamline_r):
                fa_streamlines_r[i][jj] = fa[int(x),int(y),int(z)] #fa of every point 

    
    return fa_streamlines_r, fa_streamlines_l

def plot_fa_along_streamlines(fa_streamlines_r, fa_streamlines_l, BUNDLE_NAME,points_r, points_l):
    """
    Plots the FA values along 2 bundles
    """
    fa_along_streamlines_r = np.mean(fa_streamlines_r, axis=0)
    fa_along_streamlines_l = np.mean(fa_streamlines_l, axis=0)

    legend_elements = [Line2D([0], [0], color='#73b504', lw=4, label='Right streamline'),
                        Line2D([0], [0], color='#800080', lw=4, label='Left streamline')]
  
    plt.plot(points_r, fa_along_streamlines_r,  color='#73b504')
    plt.plot(points_l, fa_along_streamlines_l,  color='#800080')
    plt.legend(handles=legend_elements, loc='upper right')
    plt.ylabel('FA')
    plt.xlabel('Streamline index')
    plt.title(f'FA mean projection along left and right streamlines, bundle {BUNDLE_NAME}')
    plt.show()

def plot_histogram(fa_streamlines_r, fa_streamlines_l, BUNDLE_NAME, viz=True):
    """
    Plots the histogram of the FA along two bundles with 100 streamlines approximation
    """
    fa_histogram_r = approximate(np.mean(fa_streamlines_r, axis=1))
    fa_histogram_l = approximate(np.mean(fa_streamlines_l, axis=1))
    if viz:
        plt.bar(np.arange(len(fa_histogram_l)),fa_histogram_l, align='edge', width=1.0, color='#800080',alpha=0.5)
        plt.bar(np.arange(len(fa_histogram_r)),fa_histogram_r, align='edge', width=1.0, color='#73b504',alpha=0.5)

        legend_elements = [Line2D([0], [0], color='#73b504', lw=4, label='Right streamline'),
                            Line2D([0], [0], color='#800080', lw=4, label='Left streamline')]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title(f'FA histogram along left and right streamlines, bundle {BUNDLE_NAME}')

        plt.show()
    else:
        return fa_histogram_r,fa_histogram_l

def length_plotting(bundle_l, bundle_r, BUNDLE_NAME):
    """
    Length projection of the left and right streamlines
    """
    lengths_l, lengths_r = list(length(bundle_l)), list(length(bundle_r))
    legend_elements = [Line2D([0], [0], color='#73b504', lw=4, label='Right streamline'),
                        Line2D([0], [0], color='#800080', lw=4, label='Left streamline')]

    fig_hist, ax = plt.subplots(1)
    ax.hist(lengths_l, color='#800080', alpha=0.7)
    ax.hist(lengths_r, color='#73b504', alpha=0.7)

    plt.legend(handles=legend_elements, loc='upper right')
    ax.set_xlabel('Length')
    ax.set_ylabel('Count')
    plt.title(f'Length projection of the streamlines from the {BUNDLE_NAME} bundle')
    plt.show()

def get_shape_similarity_score(BUNDLE_NAME, fa_affine):
    """
    Calculates the shape similarity score between left and right bundle

    Returns
    -------
    score: shape similarity score
    """
    bundle_l, bundle_r = load_bundle(BUNDLE_NAME + '_L.trk', fa_affine, transfrom=False), load_bundle(BUNDLE_NAME + '_R.trk', fa_affine, transfrom=False)
    flip_mat = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float64)
    bundle_r_flipped = transform_streamlines(bundle_r, flip_mat, in_place=False)
   
    no_points = min(len(bundle_l), len(bundle_r_flipped))
    bundle_r = set_number_of_points(bundle_r_flipped, no_points)
    bundle_l = set_number_of_points(bundle_l, no_points)

    return bundle_shape_similarity(bundle_r, bundle_l, rng=np.random.RandomState(), clust_thr=[0], threshold=10)

def approximate(array):
    """
    Approximates an array, calculating the mean values for every 100 numbers from the array
    """
    skip_step = 100
    new_array = []
    for i in range(0,len(array), skip_step):
        if i + skip_step <= len(array):
            mean = np.mean(array[i:(i + skip_step)], axis=0)
        else:
            mean = np.mean(array[i:], axis=0)
        new_array.append(mean)

    return np.array(new_array)

def vizualization(bundle_r, bundle_l, labels, affine):
    scene = window.Scene()
    if bundle_r:
        right_actor = actor.line(bundle_r,colormap.line_colors(bundle_r), opacity=0.5)
        scene.add(right_actor)
    if bundle_l:
        left_actor = actor.line(bundle_l,(128, 0, 128), opacity=0.5)
        scene.add(left_actor)
    surface_opacity = 0.9
    surface_color = [1, 1, 0]
    seedroi_actor = actor.contour_from_roi(labels, affine,
                                       surface_color, surface_opacity)
    scene.add(seedroi_actor)
    window.show(scene)

def use_HCP(bundle_name, gfa_file):
    BUNDLE_NAME = bundle_name
    fa, fa_affine = load_nifti_format(gfa_file)
    bundle_L, bundle_R = load_bundle(BUNDLE_NAME + '_L.trk', fa_affine), load_bundle(BUNDLE_NAME + '_R.trk', fa_affine)

    points_l = np.linspace(0, len(bundle_L), num=100)
    points_r = np.linspace(0, len(bundle_R), num=100)
    fa_streamlines_r, fa_streamlines_l = get_fa_along_streamlines(bundle_L, bundle_R, fa)


    similarity_score = get_shape_similarity_score(BUNDLE_NAME, fa_affine)
    right_length, left_length = np.sum(list(length(bundle_L))), np.sum(list(length(bundle_R)))

    print(f'\n\n\t\tSymmetry related features extracted, bundle {args.bundle_name}')
    print(f'Shape similarity score: {similarity_score}')
    print(f'Total length of the right streamlines: {right_length} [mm]')
    print(f'Total length of the left streamlines: {left_length} [mm]')
    print('\n\n')

                #PLOTTING
    plot_fa_along_streamlines(fa_streamlines_r, fa_streamlines_l, BUNDLE_NAME, points_r, points_l)
    plot_histogram(fa_streamlines_r, fa_streamlines_l, BUNDLE_NAME)
    length_plotting(bundle_L, bundle_R, BUNDLE_NAME)

    #             #VIZUALIZATION
    vizualization(bundle_L, bundle_R)

def load_subject(subject_index, viz=True):
    print('Loading and downsalping the data...')
    file_name = f'{subject_index}_nii4D_RAS.nii.gz'
    data, affine = load_nifti_format(file_name, 'SUBJECTS', downsample=True)
    binary_mask, _ = load_nifti_format(f'{subject_index}_dwi_binary_mask.nii.gz', 'SUBJECTS', downsample=True)
    lables, _ = load_nifti_format(f'{subject_index}_chass_symmetric3_labels_RAS_lr_ordered.nii.gz', 'SUBJECTS', downsample=True)
    bvals, bvecs = np.loadtxt(f'SUBJECTS/{subject_index}_bvals_fix.txt'), np.loadtxt(f'SUBJECTS/{subject_index}_bvec_fix.txt')
    bvec_orient = [-2,1,3]
    bvec_sign = bvec_orient/np.abs(bvec_orient)
    bvecs = np.c_[bvec_sign[0]*bvecs[:, np.abs(bvec_orient[0])-1], bvec_sign[1]*bvecs[:, np.abs(bvec_orient[1])-1],
              bvec_sign[2]*bvecs[:, np.abs(bvec_orient[2])-1]]


    mask = (lables == 51)

    grad_tab = gradient_table(bvals, bvecs)

    print('Building the CSA model')
    csa_model = CsaOdfModel(grad_tab, sh_order=6)
    print('Extracting the peaks')
    csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.25,
                             min_separation_angle=25,
                             mask=binary_mask)
    fa = csa_peaks.gfa
    # plt.imshow(fa[:,:,fa.shape[-1] // 2], cmap='gray')
    # plt.show()
    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25) 
    seeds = utils.seeds_from_mask(binary_mask, affine, density=1)
    print('Generating the streamlines')
    streamline_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                     affine=affine, step_size=0.1)
    all_streamlines = Streamlines(streamline_generator)
    stramlines = []
    ratio = 10
    
    for indx, streamline in enumerate(all_streamlines):
        if indx % ratio == 0:
            stramlines.append(streamline)
    print('Extracting targeted streamlines')
    target_streamlines =  Streamlines(utils.target(stramlines, affine, mask))
    if viz:
        vizualization(target_streamlines, None, mask, affine)

    return stramlines, fa
    

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--use_subjects', type=str, default='yes') 
    parser.add_argument('--bundle', type=str, default='AST')
    parser.add_argument('--gfa', type=str, default='HCP842_gfa.nii')

    args = parser.parse_args()

    if args.use_subjects == 'no': 
        use_HCP(args.bundle, args.gfa)

    else:
        stramlines, fa = load_subject('N57709', viz=True)

        




