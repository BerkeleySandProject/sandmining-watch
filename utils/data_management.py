import datetime
from typing import List

from .schemas import ObservationPointer
from .gcp_utils import list_files_in_bucket_with_suffix, get_public_url

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

def path_to_observatation_key(path):
    observation_key:str = path.split("/")[-1]
    remove_strings = ["_rb.tif", "_bs.tif", "_annotations.geojson", "_s1.tif", "_s2.tif"]
    for remove_string in remove_strings:
        observation_key = observation_key.replace(remove_string, "")
    return observation_key

def observation_factory(gcp_client) -> List[ObservationPointer]:
    for site, annotations in get_annotations(gcp_client).items():
        for annotation_path in annotations:
            s2_l2a_path = annotations_path_to_s2_l2a(annotation_path)
            s2_l1c_path = annotations_path_to_s2_l1c(annotation_path)
            s1_path = annotations_path_to_s1(annotation_path)
            rgb_path = annotations_path_to_rgb(annotation_path)
            observation = ObservationPointer(
                uri_to_s1=get_public_url(s1_path),
                uri_to_s2=get_public_url(s2_l2a_path),
                uri_to_s2_l1c=get_public_url(s2_l1c_path),
                uri_to_rgb=get_public_url(rgb_path),
                uri_to_annotations=get_public_url(annotation_path),
                name=path_to_observatation_key(s2_l2a_path)
            )
            yield observation
