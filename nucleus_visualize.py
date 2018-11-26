import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib import lines

def plot_nucleus_bound(im, verts):
    # read image
    if isinstance(im, str): im = skio.imread(im)
    # create axes
    _, ax = plt.subplots(1, 1)
    ax.axis('off')
    # plot lines
    for i in range(len(verts)):
        l = lines.Line2D(verts[i][:,0], verts[i][:,1])
        ax.add_line(l)
    # show image
    plt.imshow(im)
    plt.show()


def plot_image(im):
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()

def multi_plot_image(imgs):
    plt.figure()
    num = len(imgs)
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(imgs[i], cmap=plt.get_cmap('gray'))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    import os
    path = 'K:/BIGCAT/Projects/Nuclei segmentation/data_processing/Tissue images'
    im1 = 'TCGA-18-5592-01Z-00-DX1.tif'
    im1 = plt.imread(os.path.join(path, im1))
    im2 = 'TCGA-21-5784-01Z-00-DX1.tif'
    im2 = plt.imread(os.path.join(path, im2))
    # multi_plot_image(im1, im2)


