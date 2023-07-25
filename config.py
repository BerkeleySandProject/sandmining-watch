from rastervision.core.data import ClassConfig

CLASS_CONFIG_BINARY_SAND = ClassConfig(
    colors=['grey', 'red'],
    names=['other', 'sandmine'],
    null_class='other'
)

# band order in _bs.tif ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'VV', 'VH']
S2_CHANNELS = range(0,12)
RGB_CHANNELS = [3, 2, 1]
IR_CHANNELS = [11, 10, 9]
DISPLAY_GROUPS = {
    "RGB": RGB_CHANNELS,
    "IR": IR_CHANNELS,
}

LABEL_FILENAME_CONFIGS = {
    'postfix_visual' : '_rgb.tif',
    'postfix_analysis' : '_bs.tif',
    #'postfix_timelapse' : '_Timelapse.MP4',
    'postfix_annotations' : '_annotations.geojson'
}

LABEL_FOLDER_SCHEMA = {
    'image' : '_bs',
    'rgb' : '_rgb',
    'timelapse' : '.',
    'annotations' : 'annotations'
}

