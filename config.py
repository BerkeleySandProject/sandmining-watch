from rastervision.core.data import ClassConfig

CLASS_CONFIG_BINARY_SAND = ClassConfig(
    colors=['grey', 'red'],
    names=['other', 'sandmine'],
    null_class='other'
)

LABEL_FILENAME_CONFIGS = {
    'postfix_visual' : '_RGB.tif',
    'postfix_analysis' : '_BS.tif',
    'postfix_timelapse' : '_Timelapse.MP4',
    'postfix_polygons' : '_Polygons.ndjson'
}

LABEL_FOLDER_SCHEMA = {
    'image' : '_bs',
    'rgb' : '_rgb',
    'timelapse' : '.',
    'polygons' : 'polygons'
}

