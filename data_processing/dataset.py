import os
import numpy as np
import skimage.io as skio
from scipy import ndimage as ndi
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle
# from nucleus_visualize import plot_image, multi_plot_image, plot_nucleus_bound

ROOTPATH = os.path.abspath('../')
DATAPATH = os.path.join(ROOTPATH, 'data_processing', 'norm images target 3')
ANOTPATH = os.path.join(ROOTPATH, 'data_processing', 'Annotations')

def get_allfile(path):
    return [os.path.join(path, f) for f in next(os.walk(path))[2]]

def get_allfolder(path):
    return [os.path.join(path, f) for f in next(os.walk(path))[1]]

def find(exp, n=0):
    indices = np.transpose(np.nonzero(exp))
    if n > 0:
        indices = indices[:n]
    return indices

def read_image(im_path):
    img = skio.imread(im_path, plugin='pil')
    return img[0] if img.shape == (4, ) else img

def get_all_mask(verts, img_shape):
    # define mask list
    bound_masks = []
    inside_masks = []
    outside_masks = []
    # get mask
    nuclei_num = len(verts)
    for i in range(nuclei_num):
        nuclei_mask = np.zeros(img_shape[:2], dtype='uint8')
        vert = np.round(verts[i]).astype('int') - 1
        t = vert.copy()
        vert[:, 0] = t[:, 1]
        vert[:, 1] = t[:, 0]
        vert[:, 0] = np.minimum(np.maximum(vert[:, 0], 0), img_shape[0] - 1)
        vert[:, 1] = np.minimum(np.maximum(vert[:, 1], 0), img_shape[1] - 1)
        vert = np.concatenate((vert, [vert[0]]), axis=0)
        for j in range(len(vert) - 1):
            sign = 1 if vert[j + 1, 1] >= vert[j, 1] else -1
            nuclei_mask[vert[j, 0], vert[j, 1] : vert[j+1, 1] + sign : sign] = 1
            sign = 1 if vert[j + 1, 0] >= vert[j, 0] else -1
            nuclei_mask[vert[j, 0] : vert[j+1, 0] + sign : sign, vert[j+1, 1]] = 1
        # bound
        structure = ndi.generate_binary_structure(2, 1)
        bound_mask = ndi.binary_dilation(nuclei_mask, structure).astype(nuclei_mask.dtype)
        bound_masks.append(bound_mask)
        # inside
        nuclei_mask = ndi.binary_fill_holes(bound_mask)
        inside_mask = nuclei_mask - bound_mask
        if inside_mask.max() == 1:
            inside_masks.append(inside_mask)
        # outside
        outside_masks.append(np.ones(img_shape[:2], dtype='uint8') - nuclei_mask)
    outside_mask = np.ones(img_shape[:2], dtype='uint8')
    for mask in outside_masks:
        outside_mask = outside_mask & mask
    return bound_masks, inside_masks, outside_mask

def generate_patches(im_path, verts, sample_per_nuclear=10, patch_size=51):
    # read image
    img = read_image(im_path)
    nuclei_num = len(verts)
    # get mask
    bound_masks, inside_masks, outside_mask = get_all_mask(verts, img.shape)
    # padding image
    r = np.int((patch_size - 1) / 2)
    img = np.pad(img, ((r, r), (r, r), (0, 0)), 'symmetric')
    print(img.shape)
    # generate dataset
    patch_shape = (patch_size, patch_size, 3)
    bound_patchs = []
    inside_patchs = []
    for mask in bound_masks:
        bound_patchs.extend(crop_patch(img, mask, sample_per_nuclear, patch_shape))
    bound_patchs = np.array(bound_patchs).astype('uint8')
    for mask in inside_masks:
        inside_patchs.extend(crop_patch(img, mask, sample_per_nuclear, patch_shape))
    inside_patchs = np.array(inside_patchs).astype('uint8')
    outside_patchs = crop_patch(img, outside_mask, 2 * nuclei_num * sample_per_nuclear, patch_shape).astype('uint8')

    return bound_patchs, inside_patchs, outside_patchs


def crop_patch(im, mask, num, patch_shape):
    indices = find(mask == 1)
    step = np.int(indices.shape[0] / num)
    if step < 1: step = 1
    indices = indices[::step]
    patchs = np.zeros((indices.shape[0],) + patch_shape)
    for i in range(indices.shape[0]):
        ix = indices[i, 0]
        iy = indices[i, 1]
        patchs[i] = im[ix : ix + patch_shape[0], iy : iy + patch_shape[0]]

    return patchs

def create_patches_from_image(im_path, patch_shape, norm=True):
    # read img
    im = read_image(im_path)
    # padding image
    r = np.int32((np.array(patch_shape) - 1) / 2)
    im_padded = np.pad(im, ((r[0], r[0]), (r[1], r[1]), (0, 0)), 'symmetric')
    # define patches
    patches = np.zeros((np.int32(im.shape[0] * im.shape[1] / 4), ) + patch_shape)
    #
    count = 0
    for i in np.arange(0, im.shape[0], 2):
        for j in np.arange(0, im.shape[1], 2):
            patches[count] = im_padded[i : i + patch_shape[0], j : j + patch_shape[1]]
            count += 1
    # normalize
    if norm:
        with open('patch_mean.pic', 'rb') as f:
            patch_mean = pickle.load(f)
            patches /= 255
            patches -= patch_mean

    return patches

def create_image_from_label(im_path, labels):
    # 0 - bound, 1 - inside, 2 - outside
    # process labels
    # labels = np.argmax(labels, axis=1)
    # labels[labels==0] = 2
    # labels[labels==1] = 1
    # labels[labels==2] = 0
    # read image
    im = read_image(im_path)
    # create mask
    im_mask = np.ones(im.shape, dtype=np.uint8)
    #
    count = 0
    for i in range(0, im.shape[0], 2):
        for j in range(0, im.shape[1], 2):
            if labels[count, 1] > 0.7:
                im_mask[i - 1 : i + 2 : 1, j - 1 : j + 2 : 1] = 255
            count += 1

    # plot_image(im_mask)

    return  im_mask


def save_patch(im_path, patch, name):
    path = os.path.join(im_path[:-4], 'patch')
    if not os.path.exists(im_path[:-4]):
        os.mkdir(im_path[:-4])
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, '{}.pic'.format(name))
    with open(path, 'wb') as f:
        pickle.dump(patch, f)

def read_patch(im_path, name):
    path = os.path.join(im_path[:-4], 'patch', '{}.pic'.format(name))
    with open(path, 'rb') as f:
        patch = pickle.load(f)

    return patch

def prepare_training_data(file_lists, permutation=False, norm=False):
    #
    patchs = []
    labels = []
    # read patchs
    for file in file_lists:
        # bound
        bound_patch = read_patch(file, 'bound')
        patchs.extend(bound_patch)
        labels.extend(np.ones((bound_patch.shape[0],)) * 0)
        # inside
        inside_patch = read_patch(file, 'inside')
        patchs.extend(inside_patch)
        labels.extend(np.ones((inside_patch.shape[0], )) * 1)
        # outside
        outside_patch = read_patch(file, 'outside')
        # outside_patch = outside_patch[0:2]
        patchs.extend(outside_patch)
        labels.extend(np.ones((outside_patch.shape[0], )) * 2)
    # to array
    patchs = np.array(patchs, dtype='float32')
    labels = np.array(labels)
    # normalize
    if norm:
        patchs /= 255
        patch_mean = np.mean(patchs, axis=0)
        patchs -= patch_mean
        if not os.path.exists('patch_mean.pic'):
            with open('patch_mean.pic', 'wb') as f:
                pickle.dump(patch_mean, f)
    # permutation
    if permutation:
        np.random.seed(100)
        indices = np.random.permutation(np.arange(patchs.shape[0]))
        patchs = patchs[indices]
        labels = labels[indices]


    return patchs, labels

def save_training_data(patchs, labels):
    with open('data_processing/patchs.pic', 'wb') as f:
        pickle.dump(patchs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data_processing/labels.pic', 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_training_data():
    with open('data_processing/patchs.pic', 'rb') as f:
        patchs = pickle.load(f)
    with open('data_processing/labels.pic', 'rb') as f:
        labels = pickle.load(f)

    return patchs, labels

def read_nucleus_bound(path):
    # parse xml file
    xml_tree = ET.parse(path)
    regions = xml_tree.find('Annotation').find('Regions')
    # read vertex
    verts = []
    for reg in regions.findall('Region'):
        vert = []
        for v in reg.find('Vertices').findall('Vertex'):
            x = float(v.get('X'))
            y = float(v.get('Y'))
            vert.append([x,y])
        verts.append(np.array(vert))

    return verts


def save_as_mask(im, verts, id='image'):
    # create dir
    image_path = os.path.join(im[:-4], 'images')
    mask_path = os.path.join(im[:-4], 'masks')
    if not os.path.exists(im[:-4]):
        os.mkdir(im[:-4])
        os.mkdir(image_path)
        os.mkdir(mask_path)
    # read image
    img = skio.imread(im)
    # save image
    skio.imsave(os.path.join(image_path, id + '.png'), img)
    nuclei_num = len(verts)
    for i in range(nuclei_num):
        nuclei_mask = np.zeros(img.shape[:2], dtype='uint8')
        vert = np.round(verts[i]).astype('int') - 1
        vert = np.concatenate((vert, [vert[0]]), axis=0)
        for j in range(len(vert) - 1):
            sign = 1 if vert[j + 1, 1] >= vert[j, 1] else -1
            nuclei_mask[vert[j, 0], vert[j, 1] : vert[j+1, 1] + sign : sign] = 255
            sign = 1 if vert[j + 1, 0] >= vert[j, 0] else -1
            nuclei_mask[vert[j, 0] : vert[j+1, 0] + sign : sign, vert[j+1, 1]] = 255
        # fill the hole
        nuclei_mask = ndi.binary_fill_holes(nuclei_mask)
        nuclei_mask.dtype = 'uint8'
        nuclei_mask *= 255
        if True:
            skio.imsave(os.path.join(mask_path, id + '_mask_{}'.format(i) + '.png'), nuclei_mask)

if __name__ == '__main__':
    imgs_path = get_allfile(DATAPATH)[0:2]
    anots_path = get_allfile(ANOTPATH)[0:2]
    num = len(imgs_path)
    import time
    t1 = time.time()
    for im_path, anot_path in zip(imgs_path, anots_path):
        verts = read_nucleus_bound(anot_path)
        # plot_nucleus_bound(im_path, verts)
        # bp, ip, op = generate_patches(im_path, verts)
        # save_patch(im_path, bp, 'bound')
        # save_patch(im_path, ip, 'inside')
        # save_patch(im_path, op, 'outside')


    # patchs, labels = prepare_training_data(imgs_path)
    print(time.time()-t1)
