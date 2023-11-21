from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import random

from rastervision.pytorch_learner.utils import color_to_triple
from rastervision.pytorch_learner import SemanticSegmentationVisualizer

from project_config import CLASS_CONFIG, RGB_BANDS
from skimage import exposure
from matplotlib.colors import ListedColormap
from typing import List
from rastervision.pytorch_learner.dataset import GeoDataset

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from shapely.geometry import Polygon
import numpy as np
from skimage import exposure

class Visualizer():
    """
    This class solves the problem of knowing which channels in an images correspons to RGB.

    Context:
    - Our earth engine exports (*_s2.tif) contains a subset of channels from Sentinel 2.
    - In our rastervision pipeline, we again (can) specify a subset of channels to use.
      The remaining channels are ignored. We refer to this selection of channels as "s2_channels".
      The reference is the channels in our earth engine export.
    
    """
    def __init__(self, s2_channels):
        self.rgb_channels = self.infer_rbg_channels(s2_channels)

    @classmethod
    def infer_rbg_channels(cls, s2_channels):
        rgb_band_idx = [e.value for e in RGB_BANDS]
        if s2_channels is None:
            return rgb_band_idx
        else:
            return [s2_channels.index(idx) for idx in rgb_band_idx]
    

    def rgb_from_bandstack(self, image):
        # apply automatic contrast selection
        p2, p98 = np.percentile(image[:,:,self.rgb_channels], (2, 98))
        image_rescale = exposure.rescale_intensity(image[:,:,self.rgb_channels], in_range=(p2, p98))
        return np.clip(image_rescale, 0, 1.)


    def show_windows(self, img, windows, title=''):
        rgb_img = self.rgb_from_bandstack(img)
        fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
        ax.matshow(rgb_img)
        
        ax.axis('off')
        # draw windows on top of the image
        for w in windows:
            p = patches.Polygon(w.to_points(), color='r', linewidth=1.5, fill=False)
            ax.add_patch(p)

        ax.autoscale()
        ax.set_title(title)
        plt.show()

    def show_rgb_with_labels(self, img, label_img):
        rgb_img = self.rgb_from_bandstack(img)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.tight_layout(w_pad=-2)
        #enhance the contrast automatically
        rgb_img = exposure.equalize_adapthist(rgb_img, clip_limit=0.03)
        ax1.matshow(rgb_img)
        ax1.axis('off')
        cmap = ListedColormap(CLASS_CONFIG.color_triples)
        # cmap = get_default_cmap()
        ax2.imshow(label_img, cmap=cmap)
        ax2.axis('off')
        plt.show()

    def show_rgb_overlay_labels(self, img, label_img):
        rgb_img = self.rgb_from_bandstack(img)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        fig.tight_layout(w_pad=-2)
        #enhance the contrast automatically
        # rgb_img = exposure.equalize_adapthist(rgb_img, clip_limit=0.03)
        rgb_img = exposure.adjust_gamma(rgb_img)
        ax1.imshow(rgb_img)
        ax1.axis('off')
        cmap = ListedColormap(CLASS_CONFIG.color_triples)
        #convert the label image such that any pixels with value 0 are transparent
        label_img = np.ma.masked_where(label_img == 0, label_img)
        ax1.imshow(label_img, cmap=cmap, zorder=1)
        legend_items = [
        patches.Patch(facecolor=cmap(i), edgecolor='black', label=cname)
        for i, cname in enumerate(CLASS_CONFIG.names)]
        ax1.legend(
            handles=legend_items,
            ncol=len(CLASS_CONFIG),
            loc='upper center',
            fontsize=14,
            bbox_to_anchor=(0.5, 0))
        plt.show()

    
def to_rgb(img):
    if img.shape[2] == 3:
        return img
    else:
        img_rgb = img[:,:,RGB_BANDS]
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
    
def get_cmap_from_class_colors(class_colors):
    colors = [
        color_to_triple(c) if isinstance(c, str) else c
        for c in class_colors
    ]
    colors = np.array(colors) / 255.
    cmap = mcolors.ListedColormap(colors)
    return cmap

def get_default_cmap(cvals=[0,  1], colors=CLASS_CONFIG.colors):
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    return cmap
    
def show_rgb_with_labels(img, label_img):
    # img = to_rgb(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(w_pad=-2)
    ax1.matshow(img)
    ax1.axis('off')
    cmap = get_default_cmap()
    ax2.imshow(label_img, cmap=cmap)
    ax2.axis('off')
    plt.show()

def show_rgb_labels_preds(img, labels, predictions, title="", show=False):
    img = to_rgb(img)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    fig.tight_layout(w_pad=-2)
    ax1.matshow(img)
    cmap = get_default_cmap()
    ax2.imshow(labels, cmap=cmap)
    ax3.imshow(predictions, cmap=cmap)
    ax1.set_title(title)
    ax2.set_title('Labels')
    ax3.set_title('Predictions')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    if show:
        plt.show()
    else:
        return fig
    
def show_predictions(predictions, show=False):
    cmap = get_default_cmap()
    fig = plt.imshow(predictions, cmap=cmap)
    plt.axis('off')
    if show:
        plt.show()
    else:
        return fig

    
def show_labels(img, class_config=CLASS_CONFIG):
    fig, ax = plt.subplots(figsize=(10, 10))
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




def display_aoi(raster_source, aoi_polygons):
    img = raster_source[:, :]

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
    ax.imshow(img)

    for aoi in aoi_polygons:
        p = mpatches.Polygon(
            np.array(aoi.exterior.coords), color='turquoise', linewidth=1, fill=False)
        ax.add_patch(p)

    plt.show()

def show_windows(img, windows, title='', aoi_polygons=[]):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')
    # draw windows on top of the image
    for w in windows:
        p = patches.Polygon(w.to_points(), color='y', linewidth=0.8, fill=False)
        ax.add_patch(p)

    for aoi in aoi_polygons:
        p = mpatches.Polygon(
            np.array(aoi.exterior.coords), color='tab:cyan', linewidth=1, fill=False, linestyle='--')
        ax.add_patch(p)
    ax.autoscale()
    ax.set_title(title)
    
    plt.show()


from .rastervision_pipeline import ThreeClassSemanticSegmentationRandomWindowGeoDataset, ThreeClassSemanticSegmentationSlidingWindowGeoDataset
def visualize_dataset(ds_list: List[GeoDataset]):
    for ds in ds_list:
        rgb_band_idx = [e.value for e in RGB_BANDS]
        img_rgb = raster_source_to_rgb(ds.scene.raster_source)

        if isinstance(ds, ThreeClassSemanticSegmentationRandomWindowGeoDataset):
            title = f"{ds.scene.id}, N={ds.max_windows}"
            windows = [ds.sample_window() for _ in range(ds.max_windows)]
        
        elif isinstance(ds, ThreeClassSemanticSegmentationSlidingWindowGeoDataset):
            title = f"{ds.scene.id}, N={len(ds.windows)}"
            windows = ds.windows
        else:
            raise ValueError("Unexpected type of dataset")
        
        show_windows(img_rgb, windows, title=title, aoi_polygons=ds.scene.aoi_polygons)


def raster_source_to_rgb(raster_source):
    rgb_band_idx = [e.value for e in RGB_BANDS]
    img = raster_source.get_raw_chip(raster_source.extent)
    img = img[:,:,rgb_band_idx]
    img = img / 3500
    """
    Alternative:
    p2, p98 = np.percentile(image[:,:,self.rgb_channels], (2, 98))
    image_rescale = exposure.rescale_intensity(image[:,:,self.rgb_channels], in_range=(p2, p98))
    """
    return np.clip(img, 0, 1)
