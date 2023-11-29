from enum import Enum

from rastervision.core.data import ClassConfig
# CLASS_NAME = 'sandmine'
# CLASS_CONFIG = ClassConfig(
#     colors=['grey', 'red'],
#     names=['other', CLASS_NAME],
#     null_class='other'
# ) 
CLASS_NAME = 'other'
# CLASS_CONFIG = ClassConfig(
#     colors=['grey', 'red', 'maroon', 'purple', 'blue'],
#     names=['other', '25', '50', '75', '100'],
#     null_class='other'
# ) 
CLASS_CONFIG = ClassConfig(
    colors=["grey", "red"],
    names=["other", "sandmine"],
    null_class="other"
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
    # The value indicates the channel idx in our GCP export of Sentinel-2 L2A data
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
    B10 = -99 # we did not not export this for L2A
    B11 = 10
    B12 = 11

RGB_BANDS = [S2Band.B4, S2Band.B3, S2Band.B2]
#IR_CHANNELS = [11, 10, 9]
# DISPLAY_GROUPS = {
#     "RGB": RGB_BANDS,
#     "IR": IR_CHANNELS,
# }

# To use hard labels set to True
# If false, uses soft labels
HARD_LABELS = True

# Labelbox
LABELBOX_PROJECT_ID = "cllbeyixh0bxt07uxfvg977h3"

# Google Cloud Platform
GCP_PROJECT_NAME = "gee-sand"
# BUCKET_NAME = "sand_mining_median"

BUCKET_NAME = "sand_mining_test"

# Weight and Biases
WANDB_PROJECT_NAME = "sandmine_detector"

#To use river AOIs set this to TRUE
USE_RIVER_AOIS = True

#Use width of buffer for AOIs
RIVER_BUFFER_M = '1000m'

#Where the annotation json (master list of obseravations) is stored
# find the file path by looking for the latest version and date in the file name
# e.g. annotations_json_v0.1_2021-04-08.json

DATASET_JSON_PATH = 'dataset/dataset_v0.1.1-test.json'

RIVER_NETWORKS_DIR = '/data/sand_mining/rivers/'

# Number of pixels to discard at the edge of every window's prediction (in inference/evaluation mode)
# Sliding window stride will depend on this value
# See https://www.notion.so/andoshah/Inference-strategy-4da86e75cad848eeada298141ef23370?pvs=4
N_EDGE_PIXELS_DISCARD = 30
