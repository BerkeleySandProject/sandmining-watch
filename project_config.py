from enum import Enum
from experiment_configs.annotation_configs import two_class_config, three_class_config

# Annotation stuff
# Set either 2-class of 3-class config as the annotation config

# ANNO_CONFIG = two_class_config # or
ANNO_CONFIG = three_class_config

# Set global variables
CLASS_CONFIG = ANNO_CONFIG.class_config
LABELBOX_PROJECT_ID = ANNO_CONFIG.labelbox_project_id

if ANNO_CONFIG.num_classes == 2:
    CLASS_NAME = "sandmine"
else:
    # this should trigger an error wherever 3-class annotations rely on this
    CLASS_NAME = "other"


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
    B10 = -99  # we did not not export this for L2A
    B11 = 10
    B12 = 11


RGB_BANDS = [S2Band.B4, S2Band.B3, S2Band.B2]
# IR_CHANNELS = [11, 10, 9]
# DISPLAY_GROUPS = {
#     "RGB": RGB_BANDS,
#     "IR": IR_CHANNELS,
# }

# To use hard labels set to True
# If false, uses soft labels
HARD_LABELS = True


# Storage stuff

# Google Cloud Platform
GCP_PROJECT_NAME = "gee-sand"
BUCKET_NAME = "sand_mining_median"

# BUCKET_NAME = "sand_mining_test"

# Weight and Biases
WANDB_PROJECT_NAME = "DS-v03"

# To use river AOIs set this to TRUE
USE_RIVER_AOIS = True

# Use width of buffer for AOIs
RIVER_BUFFER_M = "1000m"

# Where the annotation json (master list of obseravations) is stored
# find the file path by looking for the latest version and date in the file name
# e.g. annotations_json_v0.1_2021-04-08.json

# DATASET_JSON_PATH = 'dataset/dataset_v0.2_2023-11-26.json'
DATASET_JSON_PATH = "dataset/" + "dataset_v0.3.1-sr-seed42-remove-no-l1c.json"

RIVER_NETWORKS_DIR = "/data/sand_mining/rivers/"

NUM_EPOCHS = 30

RUN_NAME = "Satlas-A-SR-Seed42"

# Number of pixels to discard at the edge of every window's prediction (in inference/evaluation mode)
# Sliding window stride will depend on this value
# See https://www.notion.so/andoshah/Inference-strategy-4da86e75cad848eeada298141ef23370?pvs=4
N_EDGE_PIXELS_DISCARD = 30
