import datetime
from typing import List

from .schemas import ObservationPointer
from .gcp_utils import list_files_in_bucket_with_suffix, list_files_in_bucket_with_prefix, get_public_url
from project_config import RIVER_BUFFER_M

def get_date_from_key(key: str):
    # Example: From "Sone_Rohtas_84-21_24-91_2022-03-01_rgb_jhjhfhdd" this function returns a the datetime 2022-03-01 00:00:00

    year, month, day = key.split("_")[4].split("-")
    # We'd prefer to set a datetime.date instead of a datetime.datetime,
    # but Labelbox doesn't accept a date. Therefore we set a timestamp.
    return datetime.datetime(int(year), int(month), int(day))

def get_location_from_key(key: str):
    # Example: From "Sone_Rohtas_84-21_24-91_2022-03-01_rgb.tif" this function returns Sone_Rohtas_84-21_24-91
    key_splitted = key.split("_")
    location = "_".join(key_splitted[:4])
    return location

def get_annotation_path(key):
    splitted = key.split("_")
    observation = "_".join(splitted[:4])
    image = "_".join(splitted[:5])
    # HACK. Need this to make annotion export to GCP work.
    # But this hack should not need to be necessary.
    # TODO: Fix problem at its root and remove the hack
    #path = f"labels/{observation}/annotations/{image}_annotations.geojson"
    path = f"labels/{observation}_median/annotations/{image}_annotations.geojson"
    return path

def get_river_path(key, buffer_m):
    splitted = key.split("_")
    observation = "_".join(splitted[:4])
    image = "_".join(splitted[:5])
    path = f"labels/{observation}_median/rivers/{observation}_rivers_{buffer_m}m.geojson"
    return path

def get_annotations(gcp_client):
    # Returns dictionary where key is location (e.g. "Ken_Banda_80-35_25-68")
    # and value is list of paths to _annotations.geojson
    all_annotations = list_files_in_bucket_with_suffix(gcp_client, "_annotations.geojson")
    annotations_dict = {}
    for annotation in all_annotations:
        location = annotation.split("/")[1]
        if location in annotations_dict:
            annotations_dict[location].append(annotation)
        else:
            annotations_dict[location] = [annotation]
    return annotations_dict

def annotations_path_to_s2_l2a(path:str):
    return path.replace("annotations", "s2").replace(".geojson", ".tif")

def annotations_path_to_s2_l1c(path:str):
    return path.replace("annotations", "s2_l1c").replace(".geojson", ".tif")

def annotations_path_to_s1(path:str):
    return path.replace("annotations", "s1").replace(".geojson", ".tif")

def annotations_path_to_rgb(path:str):
    return path.replace("annotations", "rgb").replace(".geojson", ".tif")

def annotations_path_to_rivers(path:str, buffer:str):
    path = path.replace("annotations", "rivers").replace(".geojson", f"_{buffer}.geojson")
    # now it still has the date in, strip the date
    path = path.split("_")
    # remove the 3rd last element
    path.pop(-3)
    # join the rest
    path = "_".join(path)
    return path

def global_key_to_long_lat_date(global_key:str):
    """Parse the global key and return the latitude and longitude of the centroid of the observation
       returns a list of two floats, latitude and longitude
    """
    # Example: From "Sone_Rohtas_84-21_24-91_2022-03-01_rgb.tif" this function returns 24.91, 84.21
    long_lat = global_key.split("_")[2:4]
    #now split lat_long into lat and long using the "-" as the separator between integer and decimal parts
    long_lat = [x.split("-") for x in long_lat]
    #now create a single floating point number from the two parts
    long_lat = [float(x[0]) + float(x[1])/100 for x in long_lat]

    date = global_key.split("_")[4]

    return long_lat, date




def path_to_observatation_key(path):
    observation_key:str = path.split("/")[-1]
    remove_strings = ["_rb.tif", "_bs.tif", "_annotations.geojson", "_s1.tif", "_s2.tif"]
    for remove_string in remove_strings:
        observation_key = observation_key.replace(remove_string, "")
    return observation_key

def _observation_factory_old(gcp_client, river_buffer='1000m') -> List[ObservationPointer]:
    """
    This functions yields all annoted observations in the GCP bucket.
    """
    for site, annotations in get_annotations(gcp_client).items():
        for annotation_path in annotations:
            s2_l2a_path = annotations_path_to_s2_l2a(annotation_path)
            s2_l1c_path = annotations_path_to_s2_l1c(annotation_path)
            s1_path = annotations_path_to_s1(annotation_path)
            rgb_path = annotations_path_to_rgb(annotation_path)
            rivers_path = annotations_path_to_rivers(annotation_path, river_buffer)
            observation = ObservationPointer(
                uri_to_s1=get_public_url(s1_path),
                uri_to_s2=get_public_url(s2_l2a_path),
                uri_to_s2_l1c=get_public_url(s2_l1c_path),
                uri_to_rgb=get_public_url(rgb_path),
                uri_to_annotations=get_public_url(annotation_path),
                uri_to_rivers=get_public_url(rivers_path),
                name=path_to_observatation_key(s2_l2a_path)
            )
            yield observation

def generate_observation_pointer(annotation_path:str, global_key:str, cluster_id:int) -> ObservationPointer:
    """
    This function generates an ObservationPointer from an annotation path.
    """
    s2_l2a_path = annotations_path_to_s2_l2a(annotation_path)
    s2_l1c_path = annotations_path_to_s2_l1c(annotation_path)
    s1_path = annotations_path_to_s1(annotation_path)
    rgb_path = annotations_path_to_rgb(annotation_path)
    rivers_path = annotations_path_to_rivers(annotation_path, RIVER_BUFFER_M)
    long_lat, date = global_key_to_long_lat_date(global_key)
    observation = ObservationPointer(
        uri_to_s1=get_public_url(s1_path),
        uri_to_s2=get_public_url(s2_l2a_path),
        uri_to_s2_l1c=get_public_url(s2_l1c_path),
        uri_to_rgb=get_public_url(rgb_path),
        uri_to_annotations=get_public_url(annotation_path),
        uri_to_rivers=get_public_url(rivers_path),
        name=path_to_observatation_key(s2_l2a_path),
        cluster_id=cluster_id,
        latitude=long_lat[1],
        longitude=long_lat[0],
        date=date
    )
    return observation

def observation_factory(dataset_json) -> List[ObservationPointer]:
    """
    This functions yields all annoted observations in the GCP bucket, prestored in the dataset_*.json file
    """ 
    #iterate over json file and create ObservationPointer

    for observation in dataset_json:
        observation = ObservationPointer(
            uri_to_s1           = observation['uri_to_s1'],
            uri_to_s2           = observation['uri_to_s2'],
            uri_to_s2_l1c       = observation['uri_to_s2_l1c'],
            uri_to_rgb          = observation['uri_to_rgb'],
            uri_to_annotations  = observation['uri_to_annotations'],
            uri_to_rivers       = observation['uri_to_rivers'],
            name                = observation['name'],
            cluster_id          = observation['cluster_id'],
            latitude            = observation['latitude'],
            longitude           = observation['longitude'],
            date                = observation['date']
        )
        yield observation


def all_observations_for_location(gcp_client, location_key, river_buffer='1000m') -> List[ObservationPointer]:
    """
    This functions yields all data for a location WITHOUT annotations
    """
    
    path_to_s2_l1c_data_in_bucket = f"labels/{location_key}_median/s2_l1c"
    for blob_to_s2_l1c in list_files_in_bucket_with_prefix(gcp_client, path_to_s2_l1c_data_in_bucket):

        # Create a dummy annotatations path. This is how the path the .geojson which annotations would look like, if it exists.
        fake_annotations_path = blob_to_s2_l1c.name.replace("s2_l1c", "annotations").replace(".tif", ".geojson")
        
        s2_l2a_path = annotations_path_to_s2_l2a(fake_annotations_path)
        s1_path = annotations_path_to_s1(fake_annotations_path)
        rgb_path = annotations_path_to_rgb(fake_annotations_path)
        rivers_path = annotations_path_to_rivers(fake_annotations_path, river_buffer)

        observation = ObservationPointer(
            uri_to_s1=get_public_url(s1_path),
            uri_to_s2=get_public_url(s2_l2a_path),
            uri_to_s2_l1c=get_public_url(blob_to_s2_l1c.name),
            uri_to_rgb=get_public_url(rgb_path),
            uri_to_annotations=None,
            uri_to_rivers=get_public_url(rivers_path),
            name=path_to_observatation_key(s2_l2a_path)
        )
        yield observation

