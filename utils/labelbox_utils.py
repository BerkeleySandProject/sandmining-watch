import os
from labelbox import Dataset, Client

from .data_management import get_date_from_key

MAPBOX_API_KEY = os.getenv('MAPBOX_API_KEY')


def check_if_dataset_exists(client: Client, dataset_name) -> bool:
    datasets = list(client.get_datasets(
        where=(Dataset.name == dataset_name)
    ))
    if len(datasets) == 0:
        return False
    elif len(datasets) == 1:
        return True
    else:
        raise ValueError("More than 1 dataset found")

def get_or_create_single_dataset(client: Client, dataset_name) -> Dataset:
    datasets = list(client.get_datasets(
        where=(Dataset.name == dataset_name)
    ))
    if len(datasets) == 0:
        print(f"Creating Dataset {dataset_name}")
        new_dataset = client.create_dataset(name=dataset_name)
        return new_dataset
    elif len(datasets) == 1:
        print(f"Found Dataset {dataset_name}")
        return datasets[0]
    else:
        raise ValueError("More than 1 dataset found")

def create_new_dataset(client: Client, dataset_name) -> Dataset:
    existing_datasets_with_name = list(client.get_datasets(
        where=(Dataset.name == dataset_name)
    ))
    if len(existing_datasets_with_name) > 0:
        raise ValueError(f"There exists already a dataset with the name {dataset_name}")
    new_dataset = client.create_dataset(name=dataset_name)
    return new_dataset

def create_data_row_dict(img_url, global_key):
    assert MAPBOX_API_KEY is not None
    row_data = {
        "tile_layer_url": img_url,
        "epsg": "EPSG4326",
        "name" : "RGB",
        "min_zoom": 4,
        "max_zoom": 20,
        "alternative_layers": [{
            "tile_layer_url": "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=" + MAPBOX_API_KEY,
            "name": "Hi-res Guidance"
          }]
    }
    data_row_dict = {
        "row_data" : row_data,
        "global_key" : global_key,
        "media_type": "TMS_GEO",
        "metadata_fields": [{
            "name": "imageDateS2",
            "value": get_date_from_key(global_key)
        }]
    }
    return data_row_dict


def get_annotation_objects_from_data_row_export(data_row_export):
    projects = list(data_row_export['projects'].values())

    # We expect that there exist only one "project"
    assert len(projects) == 1
    labels = projects[0]['labels']

    # We expect that there exist only one "labels"
    assert len(labels) == 1
    label = labels[0]

    classifications = label['annotations']['classifications']
    objects = label['annotations']['objects']
    mine_activity = get_mine_activity_flag(classifications)

    # Qualitiy check. The mine_activity flag must match whether annoted objects exist
    if len(objects) > 0 and mine_activity is False:
        raise ValueError("Quality check failed. There exists annotated objects, but mine activity flag is False")    
    if len(objects) == 0 and mine_activity is True:
        raise ValueError("Quality check failed. There exists no annotated objects, but mine activity flag is True")

    # All good. Quality check is passed.
    return objects


FEATURE_SCHEMA_ID_MINE_FLAG = "clkiqxheb0ptz0705g7xd1soo"
FEATURE_SCHEMA_ID_ANSWERS = {
    "clkiqxhec0pu007052djj8li1": False,
    "clkiqxhec0pu20705ht1qb0g9": True
}
def get_mine_activity_flag(data_row_classifications):
    for classification in data_row_classifications:
        if classification['feature_schema_id'] != FEATURE_SCHEMA_ID_MINE_FLAG:
            continue
        mine_flag_answer_id = classification['radio_answer']['feature_schema_id']
        return FEATURE_SCHEMA_ID_ANSWERS[mine_flag_answer_id]
    raise ValueError("Mine activity flag is missing.")


def get_geojson_fc_from_annotation_objects(annotation_objects):
    polygons = [o['geojson'] for o in annotation_objects]

    geojson_out = {
        "type": "FeatureCollection",
        "features": [
            {"geometry": polygon}
            for polygon in polygons
        ]
    }
    return geojson_out

def get_confidence_geojson_fc_from_annotation_objects(annotation_objects):
    geometries = [o['geojson'] for o in annotation_objects]
    confidence =[o['classifications'][0]['radio_answer']['name'] for o in annotation_objects]
    #parse confidence and convert to float after removing any spaces and "%" sign
    confidence = [float(c.replace(" ", "").replace("%", "")) for c in confidence]


    geojson_out = {}
    geojson_out['type'] = 'FeatureCollection'

    geojson_out['features'] = [ 
        { "type" : "Feature",
          "geometry": geom
        } 
        for geom in geometries
        ]
    
    #read confidence from geojson_out['features] from each 'geometries' and add to geojson_out['features]
    for i in range(len(geojson_out['features'])):
        geojson_out['features'][i]['properties'] = {'confidence': confidence[i]}



    return geojson_out
