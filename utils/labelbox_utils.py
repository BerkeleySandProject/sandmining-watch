import os
from labelbox import Dataset, Client

from .data_management import get_date_from_key

MAPBOX_API_KEY = os.getenv('MAPBOX_API_KEY')

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

    if len(labels) == 0:
        print("No labels, skipping data row.")
        return []    

    if len(labels) > 1:
        raise ValueError(f"Unexpected number of labels")
    
    return labels[0]['annotations']['objects']


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
