from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource,
    RasterioSource, RasterizedSource, Scene, SemanticSegmentationLabelSource,
    CastTransformer
)

import numpy as np

def create_scene(img_uri, label_uri, scene_id, channels, class_config):
    img_raster_source = RasterioSource(
        img_uri,
        channel_order=channels,
        allow_streaming=False,
        raster_transformers=[CastTransformer(np.uint16)]
    )

    vector_source = GeoJSONVectorSource(
        label_uri,
        img_raster_source.crs_transformer,
        ignore_crs_field=True,
        vector_transformers=[
            ClassInferenceTransformer(default_class_id=class_config.get_class_id('sandmine'))
        ]
    )

    label_raster_source = RasterizedSource(
        vector_source,
        background_class_id=class_config.null_class_id,
        extent=img_raster_source.extent
    )

    label_source = SemanticSegmentationLabelSource(
        label_raster_source, class_config=class_config
    )

    scene = Scene(id=scene_id, raster_source=img_raster_source, label_source=label_source)
    return scene
