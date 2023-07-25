from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import random

from rastervision.pytorch_learner.utils import color_to_triple
from rastervision.pytorch_learner import SemanticSegmentationVisualizer

from config import CLASS_CONFIG_BINARY_SAND, RGB_CHANNELS

def to_rgb(img):
    if img.shape[2] == 3:
        return img
    else:
        img_rgb = img[:,:,RGB_CHANNELS]
        return img_rgb

def show_image(img, title=''):
    img = to_rgb(img)
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))

    colors_mins = img.reshape(-1, img.shape[-1]).min(axis=0)
    colors_maxs = img.reshape(-1, img.shape[-1]).max(axis=0)
    img_normalized = (img - colors_mins) / (colors_maxs - colors_mins)
    ax.matshow(img_normalized)
    
    ax.axis('off')
    ax.autoscale()
    ax.set_title(title)
    plt.show()
    
def add_normalized_image_to_axis(ax, img, norm_factor=4000):
    img_normalized = img / norm_factor
    ax.matshow(img_normalized)

def get_cmap_from_class_colors(class_colors):
    colors = [
        color_to_triple(c) if isinstance(c, str) else c
        for c in class_colors
    ]
    colors = np.array(colors) / 255.
    cmap = mcolors.ListedColormap(colors)
    return cmap

def get_default_cmap(cvals=[0,  1], colors=CLASS_CONFIG_BINARY_SAND.colors):
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    return cmap
    
def show_rgb_with_labels(img, label_img):
    img = to_rgb(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=-2)
    add_normalized_image_to_axis(ax1, img)
    ax1.axis('off')
    cmap = get_default_cmap()
    ax2.imshow(label_img, cmap=cmap)
    ax2.axis('off')
    plt.show()

def show_rgb_labels_preds(img, labels, predictions):
    img = to_rgb(img)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    fig.tight_layout(w_pad=-2)
    add_normalized_image_to_axis(ax1, img)
    cmap = get_default_cmap()
    ax2.imshow(labels, cmap=cmap)
    ax3.imshow(predictions, cmap=cmap)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.show()
    
def show_windows(img, windows, title=''):
    img = to_rgb(img)
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))

    # colors_mins = img.reshape(-1, img.shape[-1]).min(axis=0)
    # colors_maxs = img.reshape(-1, img.shape[-1]).max(axis=0)
    # img_normalized = (img - colors_mins) / (colors_maxs - colors_mins)
    # ax.matshow(img_normalized)
    add_normalized_image_to_axis(ax, img)
    
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

def show_image_in_dataset(ds, class_config, display_groups, idx=None):
    visualizer = SemanticSegmentationVisualizer(
        class_names=class_config.names,
        class_colors=class_config.colors,
        channel_display_groups=display_groups,
    )
    if idx is None:
        idx = random.randint(0,len(ds)-1)
    x, y = ds[idx]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    visualizer.plot_batch(x, y, show=True)
