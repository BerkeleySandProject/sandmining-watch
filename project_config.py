from enum import Enum

from rastervision.core.data import ClassConfig
CLASS_NAME = 'sandmine'
CLASS_CONFIG = ClassConfig(
    colors=['grey', 'red'],
    names=['other', CLASS_NAME],
    null_class='other'
) 

from split import training_locations, validation_locations
def is_training(observation_key:str):
    for training_location in training_locations:
        if observation_key.startswith(training_location):
            return True
    return False

def is_validation(observation_key:str):
    for validation_location in validation_locations:
        if observation_key.startswith(validation_location):
            return True
    return False


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


RGB_BANDS = [S2Band.B4, S2Band.B3, S2Band.B2]
#IR_CHANNELS = [11, 10, 9]
# DISPLAY_GROUPS = {
#     "RGB": RGB_BANDS,
#     "IR": IR_CHANNELS,
# }


# Labelbox
LABELBOX_PROJECT_ID = "cllbeyixh0bxt07uxfvg977h3"

# Google Cloud Platform
GCP_PROJECT_NAME = "gee-sand"
# BUCKET_NAME = "sand_mining_median"

BUCKET_NAME = "sand_mining_test"

# Weight and Biases
WANDB_PROJECT_NAME = "sandmine_detector"


RIVER_NETWORKS_DIR = '/data/sand_mining/rivers/'
