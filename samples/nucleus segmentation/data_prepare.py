
from dataset import *
from scipy import ndimage as ndi
import xml.etree.ElementTree as ET

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
            nuclei_mask[vert[j, 0], vert[j, 1]: vert[j + 1, 1] + sign: sign] = 1
            sign = 1 if vert[j + 1, 0] >= vert[j, 0] else -1
            nuclei_mask[vert[j, 0]: vert[j + 1, 0] + sign: sign, vert[j + 1, 1]] = 1
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
    # generate dataset
    patch_shape = (patch_size, patch_size, 3)
    bound_patchs = []
    inside_patchs = []
    for mask in bound_masks:
        bound_patchs.extend(patches_from_mask(img, mask, sample_per_nuclear, patch_shape))
    bound_patchs = np.array(bound_patchs).astype('uint8')

    for mask in inside_masks:
        inside_patchs.extend(patches_from_mask(img, mask, sample_per_nuclear, patch_shape))
    inside_patchs = np.array(inside_patchs).astype('uint8')

    outside_patchs = patches_from_mask(img, outside_mask, 2 * nuclei_num * sample_per_nuclear, patch_shape).astype(
        'uint8')

    return bound_patchs, inside_patchs, outside_patchs

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
    with open('dataset/patchs.pic', 'wb') as f:
        pickle.dump(patchs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('dataset/labels.pic', 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

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

def convert_verts_to_mask(img_shape, anotation_path, save_path=None):
    # read verts
    verts = read_nucleus_bound(anotation_path)
    bound_masks, inside_masks, _ = get_all_mask(verts, img_shape)
    # create mask map
    mask = np.zeros(img_shape[0:2], dtype='uint8')
    bound = np.zeros(img_shape.shape[0:2], dtype='uint8')
    for in_mask in inside_masks:
        mask = mask | in_mask
        # bound += in_maks
    mask *= 255
    if save_path:
        skio.imsave(save_path, mask)