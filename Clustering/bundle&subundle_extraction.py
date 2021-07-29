from Extractor.main import Model
import numpy as np
from dipy.viz import actor, window, colormap 
import matplotlib.pyplot as plt
from dipy.tracking.streamline import set_number_of_points

brain = Model(DATASET_NAME='stanford_hardi')

affine = np.eye(4)
brain.set_new_affine(affine)
print('Building the CSA model...')
brain.build_CSA_model()
print('Generating all the streamlines..')
brain.generate_streamlines()

#target bundle: corpus collosum
print('Extracting the target streamlines...')
lables = brain.lables
corpus_collosum_mask = lables == 2
cc_stramlines = brain.get_target_streamlines(corpus_collosum_mask)

#Clustering
from dipy.segment.clustering import QuickBundles

qb = QuickBundles(threshold=7.5)
clusters = qb.cluster(cc_stramlines)


#Vizualization

biggest_bundle_index = float('-inf')
big_bundles = []

for index, cluster in enumerate(clusters):
    print('Cluster no: {}, no of the streamlines: {}'.format(index, len(cluster.indices)))
    if len(cluster.indices) >= 350:
        big_bundles.append(cluster.centroid)
    if biggest_bundle_index < len(cluster.indices):
        biggest_bundle_index = index


bundle_streamlines, k = [], 0

for i in range(len(cc_stramlines)):
    if i == clusters[biggest_bundle_index].indices[k]:
        bundle_streamlines.append(cc_stramlines[i])

        if k < len(clusters[biggest_bundle_index].indices) - 1:
            k+=1
        else:
            break

# color_list = [(np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)) 
#               for n in range(len(bundle_streamlines))]


#mapping function
def map_streamlines_to_fa(fa_map, streamlines):
    fa_streamlines = []
    for i, streamline in enumerate(streamlines):
        #fa points per streamline
        fa_streamline = [fa_map[int(p[0]),int(p[1]),int(p[2])] for p in streamline]
        #mean Fa per streamline
        fa_streamline = np.mean(fa_streamline)
        fa_streamlines.append(fa_streamline)

    return fa_streamlines

color_map = map_streamlines_to_fa(brain.FA, bundle_streamlines)
color_list = [(np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)) 
              for n in range(len(clusters))]
X, Y, Z = [],[],[]
for i, streamline in enumerate(bundle_streamlines):
    #streamline  = [[x,y,z],[x,y,z],[x,y,z], ... ]
    for (x,y,z) in streamline:
        X.append(x)
        Y.append(y)
        Z.append(z)

x_max,x_min,y_max,y_min,z_max,z_min= np.max(X), np.min(X),np.max(Y), np.min(Y),np.max(Z), np.min(Z),

target_area = brain.FA[int(x_min):int(x_max), int(y_min):int(y_max), int(z_min):int(z_max)]

# plt.imshow(target_area[:,:,target_area.shape[2] // 2 - 4].T)
# plt.show()


def vizualization():
    strline_actor = actor.line(cc_stramlines, window.colors.white, opacity=0.05)
    #bundle_actor = actor.line(bundle_streamlines, color_map, linewidth=0.4)
    surface_opacity = 0.5
    surface_color = [1, 1, 0]

    # for index, color in enumerate(color_list):
    #     print('index: {}, and color: {}'.format(index, color))

    scene = window.Scene()
    scene.add(strline_actor)
    #cene.add(bundle_actor)
    scene.add(actor.streamtube(clusters.centroids, color_list, linewidth=0.4))

    window.show(scene)

vizualization()