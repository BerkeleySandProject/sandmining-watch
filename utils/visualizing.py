from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
import matplotlib
import numpy as np

from rastervision.pytorch_learner.utils import color_to_triple

from config import CLASS_CONFIG_BINARY_SAND

def show_image(img, title=''):
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))

    colors_mins = img.reshape(-1, img.shape[-1]).min(axis=0)
    colors_maxs = img.reshape(-1, img.shape[-1]).max(axis=0)
    img_normalized = (img - colors_mins) / (colors_maxs - colors_mins)
    ax.matshow(img_normalized)
    
    ax.axis('off')
    ax.autoscale()
    ax.set_title(title)
    plt.show()
    
def add_normalized_image_to_axis(ax, img):
    colors_mins = img.reshape(-1, img.shape[-1]).min(axis=0)
    colors_maxs = img.reshape(-1, img.shape[-1]).max(axis=0)
    img_normalized = (img - colors_mins) / (colors_maxs - colors_mins)
    ax.matshow(img_normalized)

def get_cmap_from_class_colors(class_colors):
    colors = [
        color_to_triple(c) if isinstance(c, str) else c
        for c in class_colors
    ]
    colors = np.array(colors) / 255.
    cmap = mcolors.ListedColormap(colors)
    return cmap

def get_default_cmap(cvals=[0,  1], colors=["gray","red"]):
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    return cmap
    
def show_rgb_with_labels(rgb_img, label_img, class_colors=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=-2)
    add_normalized_image_to_axis(ax1, rgb_img)
    ax1.axis('off')
    if class_colors:
        cmap = get_cmap_from_class_colors(class_colors)
    else:
        cmap = get_default_cmap()
    ax2.imshow(label_img, cmap=cmap)
    ax2.axis('off')
    plt.show()
    
def show_windows(img, windows, title=''):
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))

    colors_mins = img.reshape(-1, img.shape[-1]).min(axis=0)
    colors_maxs = img.reshape(-1, img.shape[-1]).max(axis=0)
    img_normalized = (img - colors_mins) / (colors_maxs - colors_mins)
    ax.matshow(img_normalized)
    
    ax.axis('off')
    # draw windows on top of the image
    for w in windows:
        p = patches.Polygon(w.to_points(), color='r', linewidth=1, fill=False)
        ax.add_patch(p)
    ax.autoscale()
    ax.set_title(title)
    plt.show()
    
def show_labels(img, class_config=CLASS_CONFIG_BINARY_SAND):
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = mcolors.ListedColormap(class_config.color_triples)
    ax.matshow(img, cmap=cmap)
    legend_items = [
        patches.Patch(facecolor=cmap(i), edgecolor='black', label=cname)
        for i, cname in enumerate(class_config.names)]
    ax.legend(
        handles=legend_items,
        ncol=len(class_config),
        loc='upper center',
        fontsize=14,
        bbox_to_anchor=(0.5, 0))
    plt.show()
