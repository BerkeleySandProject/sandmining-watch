from enum import Enum

from rastervision.core.data import ClassConfig
CLASS_NAME = 'sandmine'
CLASS_CONFIG = ClassConfig(
    colors=['grey', 'red'],
    names=['other', CLASS_NAME],
    null_class='other'
) 


training_locations = [
    "Ganges_Patna_85-1_25-66",
    "Ganges_Patna_85-23_25-62",
    "Betwa_Jalaun_79-49_25-84",
    "Betwa_Jalaun_79-79_25-89",
    "Mahananda_UttarDinajpur_88-25_26-46",
    "Teesta_Jalpaiguri_88-6_26-84",
    "Teesta_Jalpaiguri_88-64_26-85",
    "Chambal_More_77-92_26-66",
    "Ken_Banda_80-35_25-68",
    "Kathajodi_Cuttack_85-85_20-44",
    "Tawa_Hoshangabad_77-80_22-74",
    "Sone_Patna_84-76_25-44",
    "Banas_Banaskantha_71-93_23-96",
    "Waiganga_Gondiya_80-16_21-62",
    "Waiganga_Gondiya_80-11_21-59",
    "Waiganga_Gondiya_80-03_21-55",
]
validation_locations = [
    "Bhargavi_Khordha_85-88_20-26",
    "Mahanadi_Angul_84-52_20-71",
    "Narmada_Sehore_77-32_22-56",
    "Mahananda_Jalpaiguri_88-4_26-68",
    "Sone_Rohtas_83-86_24-46",
    "Sone_Rohtas_84-21_24-91",
]

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
BUCKET_NAME = "sand_mining_median"

# Weight and Biases
WANDB_PROJECT_NAME = "sandmine_detector"


RIVER_NETWORKS_DIR = '/data/sand_mining/rivers/'
