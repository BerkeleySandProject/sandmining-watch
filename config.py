from rastervision.core.data import ClassConfig

CLASS_CONFIG_BINARY_SAND = ClassConfig(
    colors=['grey', 'red'],
    names=['other', 'sandmine'],
    null_class='other'
)

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

