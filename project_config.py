from enum import Enum

from rastervision.core.data import ClassConfig
CLASS_NAME = 'sandmine'
CLASS_CONFIG = ClassConfig(
    colors=['grey', 'red'],
    names=['other', CLASS_NAME],
    null_class='other'
) 

# Band order in _s1.tif ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
# Band order in _s2.tif ['VV', 'VH']

class S2Band(Enum):
    # The value indicates the channel idx in our GCP export
    B1 = 0
    B2 = 1
    B3 = 2
    B4 = 3
    B5 = 4
    B6 = 5
    B7 = 6
    B8 = 7
    B8A = 8
    B9 = 9
    B10 = -99 # we did not not export this
    B11 = 10
    B12 = 11


RGB_CHANNELS = [3, 2, 1]  # TODO generalize this (does not work when subset of bands are selected)
#IR_CHANNELS = [11, 10, 9]
DISPLAY_GROUPS = {
    "RGB": RGB_CHANNELS,
    #"IR": IR_CHANNELS,
}


# Labelbox
LABELBOX_PROJECT_ID = "cllbeyixh0bxt07uxfvg977h3"

# Google Cloud Platform
GCP_PROJECT_NAME = "gee-sand"
BUCKET_NAME = "sand_mining_median"

# Weight and Biases
WANDB_PROJECT_NAME = "sandmine_detector"


RIVER_NETWORKS_DIR = '/data/sand_mining/rivers/'
