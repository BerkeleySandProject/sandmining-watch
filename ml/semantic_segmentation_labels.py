from typing import (TYPE_CHECKING, Any, Iterable, List, Optional, Sequence,
                    Tuple, Union)
from abc import abstractmethod

import numpy as np
from rasterio.features import rasterize
from shapely.ops import transform

from rastervision.core.box import Box
from rastervision.core.data.label import Labels
from rastervision.core.data.label.utils import discard_prediction_edges

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        VectorOutputConfig)


from rastervision.core.data import SemanticSegmentationLabels, SemanticSegmentationSmoothLabels, SemanticSegmentationDiscreteLabels


class SemanticSegmentationLabelsCustom(SemanticSegmentationLabels):
    """Representation of Semantic Segmentation labels."""

    def __init__(self, extent: Box, num_classes: int, dtype: np.dtype):
        """Constructor.

        Args:
            extent (Box): The extent of the region to which
                the labels belong, in global coordinates.
            num_classes (int): Number of classes.
        """
        self.extent = extent
        self.num_classes = num_classes
        self.ymin, self.xmin, self.width, self.height = extent.to_xywh()
        self.dtype = dtype

    def _to_local_coords(self,
                         window: Union[Box, Tuple[int, int, int, int]]) -> Box:
        """Convert window to extent coordinates.

        Args:
            window (Union[Box, Tuple[int, int, int, int]]): A rastervision
                Box or a 4-tuple of (ymin, xmin, ymax, xmax).

        Returns:
            Box: Window in local coords.
        """

        return super()._to_local_coords(window)


    
    def add_predictions(self,
                    windows: Iterable['Box'],
                    predictions: Iterable[Any],
                    crop_sz: Optional[int] = None,
                    use_kernel=False) -> None:
        """Populate predictions
        Args:
            windows (Iterable[Box]): Boxes in pixel coords, specifying chips
                in the raster.
            predictions (Iterable[Any]): The model predictions for each chip
                specified by the windows.
            crop_sz (Optional[int]): Number of rows/columns of pixels from the
                edge of prediction windows to discard. This is useful because
                predictions near edges tend to be lower quality and can result
                in very visible artifacts near the edges of chips. This should
                only be used if the given windows represent a sliding-window
                grid over the scene extent with overlap between adjacent
                windows. Defaults to None.
        """

        super().add_predictions(windows, predictions, crop_sz=crop_sz) # -> calls SemanticSegmentationSmoothLabelsCustom.addWindow()

        # if crop_sz is not None:
        #     windows, predictions = discard_prediction_edges(
        #         windows, predictions, crop_sz)
        # # If predictions is tqdm-wrapped, it needs to be the first arg to zip()
        # # or the progress bar won't terminate with the correct count.
        # for prediction, window in zip(predictions, windows):
        #     self[window] = prediction  # -> calls SemanticSegmentationSmoothLabelsCustom.addWindow()

        # print ("Done adding predictions")

    

    @classmethod
    def from_predictions(cls,
                         windows: Iterable['Box'],
                         predictions: Iterable[Any],
                         extent: Box,
                         num_classes: int,
                         crop_sz: Optional[int] = None,
                         apply_kernel: bool = True,
                         tile_size: int = 160,
                         sigma: float = 5.
                         ) -> 'SemanticSegmentationSmoothLabelsCustom': #By putting it in quotes, you can let the interpreter know that the class is defined later in the file
        """Instantiate from windows and their corresponding predictions.

        Args:
            windows (Iterable[Box]): Boxes in pixel coords, specifying chips
                in the raster.
            predictions (Iterable[Any]): The model predictions for each chip
                specified by the windows.
            extent (Box): The extent of the region to which the labels belong,
                in global coordinates.
            num_classes (int): Number of classes.
            crop_sz (Optional[int]): Number of rows/columns of pixels from the
                edge of prediction windows to discard. This is useful because
                predictions near edges tend to be lower quality and can result
                in very visible artifacts near the edges of chips. This should
                only be used if the given windows represent a sliding-window
                grid over the scene extent with overlap between adjacent
                windows. Defaults to None.

        Returns:
            Union[SemanticSegmentationDiscreteLabels,
            SemanticSegmentationSmoothLabels]: If smooth=True, returns a
                SemanticSegmentationSmoothLabels. Otherwise, a
                SemanticSegmentationDiscreteLabels.
        """
        labels = SemanticSegmentationSmoothLabelsCustom.make_empty(extent=extent, num_classes=num_classes, apply_kernel=apply_kernel, tile_size=tile_size, sigma=sigma)
        labels.add_predictions(windows, predictions, crop_sz=crop_sz)
        return labels

class SemanticSegmentationSmoothLabelsCustom(SemanticSegmentationLabelsCustom):
    """Membership-scores for each pixel for each class.

    Maintains a num_classes x H x W array where value_{ijk} represents the
    probability (or some other measure) of pixel_{jk} belonging to class i.
    A discrete label array can be obtained from this by argmax'ing along the
    class dimension.
    """

    def __init__(self,
                 extent: Box,
                 num_classes: int,
                 dtype: Any = np.float16,
                 dtype_hits: Any = np.uint8,
                 apply_kernel: bool = True,
                 tile_size: int = 160,
                 sigma: float = 5.
                 ):
        """Constructor.

        Args:
            extent (Box): The extent of the region to which
                the labels belong, in global coordinates.
            num_classes (int): Number of classes.
            dtype (Any): dtype of the scores array. Defaults to np.float16.
            dtype_hits (Any): dtype of the hits array. Defaults to np.uint8.
        """
        super().__init__(extent, num_classes, dtype)

        self.pixel_scores = np.zeros(
            (self.num_classes, self.height, self.width), dtype=self.dtype)
        self.pixel_hits = np.zeros((self.height, self.width), dtype=dtype_hits)
        self.kernel_weights = np.ones((self.num_classes, self.height, self.width), dtype=self.dtype)

        self.apply_kernel = apply_kernel

        if self.apply_kernel:
            # print("Learner: Applying Kernel Smoothing with sigma: ", sigma)
            self.kernel = self.create_gaussian_distribution(tile_size, sigma)


    def __add__(self, other: 'SemanticSegmentationSmoothLabels'
                ) -> 'SemanticSegmentationSmoothLabels':
        """Merge self with other by adding pixel scores and hits."""
        if self.extent != other.extent:
            raise ValueError('Cannot add labels with unqeual extents.')

        self.pixel_scores += other.pixel_scores
        self.pixel_hits += other.pixel_hits

        return self

    def __eq__(self, other: 'SemanticSegmentationSmoothLabels') -> bool:
        if not isinstance(other, SemanticSegmentationSmoothLabels):
            return False
        if self.extent != other.extent:
            return False
        scores_equal = np.allclose(self.pixel_scores, other.pixel_scores)
        hits_equal = np.array_equal(self.pixel_hits, other.pixel_hits)

        return (scores_equal and hits_equal)

    def __delitem__(self, window: Box) -> None:
        """Reset scores and hits to zero for pixels in the window."""
        y0, x0, y1, x1 = self._to_local_coords(window)
        self.pixel_scores[..., y0:y1, x0:x1] = 0
        self.pixel_hits[..., y0:y1, x0:x1] = 0


    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_score_arr(window)
    
    def create_gaussian_distribution(self, length, sigma):
        """Create a Gaussian distribution with a given length and sigma.

        :param length: Length of the distribution array.
        :param sigma: Standard deviation of the Gaussian distribution.
        :return: Gaussian distribution array.
        """
        # x = np.linspace(-length / 2, length / 2, length)
        x = np.linspace(0, length / 2, length)
        gaussian = np.exp(-0.5 * (x / sigma) ** 2)
        # gaussian /= gaussian.sum()  # Normalize the distribution
        return 1-gaussian
    

    def create_linear_distribution(self, max_distance, length):
        """
        Create a linear distribution with a given length.

        :param length: Length of the distribution array.
        :return: Linear distribution array.
        """
        x1 = np.linspace(0, 1, max_distance, dtype=np.float16) 
        
        x2 = np.linspace(1, 0, max_distance, dtype=np.float16) 
        x_12 = np.ones(length - max_distance*2)
        x = np.concatenate((x1, x_12, x2))
        return x


    def apply_edge_based_gaussian1D(self,size, sigma=5):
        """
        Apply a Gaussian effect based on the distance from the edge of the square.

        :param size: Size of the square (width and height).
        :param sigma: Sigma of the Gaussian distribution.
        :param max_distance: Maximum distance to consider from the edge for the Gaussian effect.
        :return: Image with applied Gaussian effect.
        """
        # Create a Gaussian distribution
        distribution = self.create_gaussian_distribution(size, sigma)

        # distribution = self.create_linear_distribution(max_distance, size)

        # Create an empty square
        image = np.zeros((size, size))

        # Apply the Gaussian effect
        for i in range(size):
            for j in range(size):
                # Calculate the minimum distance from the edge
                distance_from_edge = min(i, j, size - 1 - i, size - 1 - j)
                # Sample from the Gaussian distribution based on the distance
                # gaussian_index = int(max_distance - distance_from_edge)
                gaussian_index = int(distance_from_edge)
                image[i, j] = distribution[gaussian_index]

        return image
    
    def normalize(self, image):
        """
        Normalize an image to be between 0 and 1.

        :param image: Image to normalize.
        :return: Normalized image.
        """
        image_min = image.min()
        image_max = image.max()
        return (image - image_min) / (image_max - image_min)
    
    def apply_edge_based_gaussian2D(self, size_y, size_x):
        """
        Applies an edge-based Gaussian effect to an image.

        Args:
            size_x (int): The width of the image.
            size_y (int): The height of the image.

        Returns:
            numpy.ndarray: The image with the applied Gaussian effect.
        """

        # Create an empty rectangle
        image = np.zeros((size_y, size_x))

        # Apply the Gaussian effect
        for i in range(size_y):
            for j in range(size_x):
                # Calculate the minimum distance from the edge
                distance_from_edge = min(i, j, size_y - 1 - i, size_x - 1 - j)
                # Sample from the Gaussian distribution based on the distance
                # gaussian_index = int(max_distance - distance_from_edge)

                image[i, j] = self.kernel[int(distance_from_edge)]

        return self.normalize(image) # This is needed because of the way the kernel is generated for non-square images when sigma is high 
                                    #  in those cases, the maximum value is < 1
        # return image

    
    def generate_smoothed_scores(self,
                                 num_classes,
                                 pixel_class_scores
                                 ):
        """
        Generate kernel weights based on the pixel class scores.

        :param pixel_class_scores: Pixel class scores.
        :return: smoothed scores
        """

        smoothed = np.zeros((num_classes, pixel_class_scores.shape[0], pixel_class_scores.shape[1]))
        applied_kernel = self.apply_edge_based_gaussian2D(pixel_class_scores.shape[0], pixel_class_scores.shape[1])

       
        for i in range(num_classes):

            if self.apply_kernel:
                smoothed[i] = applied_kernel  * pixel_class_scores
            else:
                smoothed[i] = pixel_class_scores

        return smoothed, applied_kernel

    

    def add_window(self, window: Box, pixel_class_scores: np.ndarray, display=False) -> None:
        # self.extent coords
        window_dst = window.intersection(self.extent)
        dst_yslice, dst_xslice = window_dst.to_slices()
        # pixel_class_scores coords
        window_src = window_dst.to_global_coords(
            self.extent).to_local_coords(window)
        src_yslice, src_xslice = window_src.to_slices()

        pixel_class_scores = pixel_class_scores.astype(self.dtype)
        pixel_class_scores = pixel_class_scores[..., src_yslice, src_xslice]

        num_classes = self.pixel_scores.shape[0]

        smoothed_scores, kernel = self.generate_smoothed_scores(num_classes, pixel_class_scores)


        self.pixel_scores[..., dst_yslice, dst_xslice] += smoothed_scores
        self.pixel_hits[dst_yslice, dst_xslice] += 1
        self.kernel_weights[..., dst_yslice, dst_xslice] += kernel
        # print ("SemanticSegmentationSmoothLabelsCustom add_window")
        # print (dst_yslice, dst_xslice ," +=", src_yslice, src_xslice)
        # print (self.pixel_scores.shape, pixel_class_scores.shape, kernel_weights.shape)
        if display:
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            axs[0].imshow(pixel_class_scores, vmax=1, vmin=0)
            axs[0].set_title('Predictions')
            p=axs[1].imshow(self.kernel_weights[0, dst_yslice, dst_xslice], vmin=0)
            axs[1].set_title('Kernel Weights')
            axs[2].imshow(smoothed_scores[0], vmax=1, vmin=0)
            axs[2].set_title('Smoothed')
            axs[3].imshow(self.pixel_scores[0,dst_yslice, dst_xslice], vmin=0)
            #move title to under the plots

            axs[3].set_title('Added')
            fig.colorbar(p, ax=axs[1], shrink=0.5)
            plt.text(0.5, -0.4,f'{window}', ha='center', va='center', transform=axs[1].transAxes)
            # plt.colorbar(shrink=0.5)
            plt.show()
            
      

    def get_score_arr(self, window: Box) -> np.ndarray:
        """Get array of pixel scores."""
        y0, x0, y1, x1 = window.intersection(self.extent)
        scores = self.pixel_scores[..., y0:y1, x0:x1]  #classes:y:x
        hits = self.pixel_hits[y0:y1, x0:x1]

        if self.apply_kernel:
            avg_scores = scores / self.kernel_weights[..., y0:y1, x0:x1]
        else:
            avg_scores = scores / hits

        return avg_scores

    def get_label_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get labels as array of class IDs.

        Returns null_class_id for pixels for which there is no data.
        """
        avg_scores = self.get_score_arr(window)
        label_arr = np.argmax(avg_scores, axis=0)
        mask = np.isnan(avg_scores[0])
        return np.where(mask, null_class_id, label_arr)

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Set fill_value'th class ID's score to 1 and all others to zero."""
        class_id = fill_value
        y0, x0, y1, x1 = self._to_local_coords(window)
        h, w = y1 - y0, x1 - x0
        mask = mask[:h, :w]
        self.pixel_scores[..., y0:y1, x0:x1][..., mask] = 0
        self.pixel_scores[class_id, y0:y1, x0:x1][mask] = 1
        self.pixel_hits[y0:y1, x0:x1][mask] = 1
        print ("SemanticSegmentationSmoothLabels mask_fill")

    @classmethod
    def make_empty(cls, extent: Box,
                   num_classes: int,
                   apply_kernel: bool,
                    tile_size: int,
                    sigma: float
                    ) -> 'SemanticSegmentationSmoothLabels':
        """Instantiate an empty instance."""
        return cls(extent=extent, num_classes=num_classes, apply_kernel=apply_kernel,tile_size = tile_size, sigma = sigma)

    @classmethod
    def from_predictions(cls,
                         windows: Iterable['Box'],
                         predictions: Iterable[Any],
                         extent: Box,
                         num_classes: int,
                         crop_sz: Optional[int] = None,
                         apply_kernel: bool = True,
                         tile_size: int = 160,
                         sigma: float = 5.
                         ) -> Union['SemanticSegmentationDiscreteLabels',
                                    'SemanticSegmentationSmoothLabels']:
        labels = cls.make_empty(extent, num_classes, apply_kernel=apply_kernel, tile_size=tile_size, sigma=sigma)
        labels.add_predictions(windows, predictions, crop_sz=crop_sz)
        # print("SemanticSegmentationSmoothLabels")
        return labels

    def save(self,
             uri: str,
             crs_transformer: 'CRSTransformer',
             class_config: 'ClassConfig',
             tmp_dir: Optional[str] = None,
             save_as_rgb: bool = False,
             discrete_output: bool = True,
             smooth_output: bool = True,
             smooth_as_uint8: bool = False,
             rasterio_block_size: int = 512,
             vector_outputs: Optional[Sequence['VectorOutputConfig']] = None,
             profile_overrides: Optional[dict] = None) -> None:
        """Save labels as rasters and/or vectors.

        If URI is remote, all files will be first written locally and then
        uploaded to the URI.

        Args:
            uri (str): URI of directory in which to save all output files.
            crs_transformer (CRSTransformer): CRSTransformer to configure CRS
                and affine transform of the output GeoTiff(s).
            class_config (ClassConfig): The ClassConfig.
            tmp_dir (Optional[str], optional): Temporary directory to use. If
                None, will be auto-generated. Defaults to None.
            save_as_rgb (bool, optional): If True, saves labels as an RGB
                image, using the class-color mapping in the class_config.
                Defaults to False.
            discrete_output (bool, optional): If True, saves labels as a raster
                of class IDs (one band). Defaults to True.
            smooth_output (bool, optional): If True, saves labels as a raster
                of class scores (one band for each class). Defaults to True.
            smooth_as_uint8 (bool, optional): If True, stores smooth class
                scores as np.uint8 (0-255) values rather than as np.float32
                discrete labels, to help save memory/disk space.
                Defaults to False.
            rasterio_block_size (int, optional): Value to set blockxsize and
                blockysize to. Defaults to 512.
            vector_outputs (Optional[Sequence[VectorOutputConfig]], optional):
                List of VectorOutputConfig's containing vectorization
                configuration information. Only classes for which a
                VectorOutputConfig is specified will be saved as vectors.
                If None, no vector outputs will be produced. Defaults to None.
            profile_overrides (Optional[dict], optional): This can be used to
                arbitrarily override properties in the profile used to create
                the output GeoTiff(s). Defaults to None.
        """
        from rastervision.core.data import SemanticSegmentationLabelStore

        label_store = SemanticSegmentationLabelStore(
            uri=uri,
            extent=self.extent,
            crs_transformer=crs_transformer,
            class_config=class_config,
            tmp_dir=tmp_dir,
            save_as_rgb=save_as_rgb,
            discrete_output=discrete_output,
            smooth_output=smooth_output,
            smooth_as_uint8=smooth_as_uint8,
            rasterio_block_size=rasterio_block_size,
            vector_outputs=vector_outputs)
        label_store.save(self, profile=profile_overrides)



        

# Define parameters
square_size = 100
sigma_value = 10
max_distance_value = 50



    



