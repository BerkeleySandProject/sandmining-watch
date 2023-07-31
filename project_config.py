# Rastervision Class Config
from rastervision.core.data import ClassConfig
CLASS_NAME = 'sandmine'
CLASS_CONFIG = ClassConfig(
    colors=['grey', 'red'],
    names=['other', CLASS_NAME],
    null_class='other'
) 

# Band order in _bs.tif ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'VV', 'VH']
S2_CHANNELS = range(0,12)
RGB_CHANNELS = [3, 2, 1]
IR_CHANNELS = [11, 10, 9]
DISPLAY_GROUPS = {
    "RGB": RGB_CHANNELS,
    "IR": IR_CHANNELS,
}

# Labelbox
LABELBOX_PROJECT_ID = "cljoqdjei070j0729f20hc5sx"

# Google Cloud Platform
GCP_PROJECT_NAME = "gee-sand"
BUCKET_NAME = "sand_mining"
