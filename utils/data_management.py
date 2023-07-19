import datetime
from typing import List

from .schemas import Split, ObservationPointer
from .gcp_utils import list_files_in_bucket_with_suffix, get_public_url

def get_date_from_key(key):
    # Example: From "Sone_Rohtas_84-21_24-91_2022-03-01_rgb.tif" this function returns a the datetime 2022-03-01 00:00:00
    year, month, day = key.split("_")[-2].split("-")
    # We'd prefer to set a datetime.date instead of a datetime.datetime,
    # but Labelbox doesn't accept a date. Therefore we set a timestamp.
    return datetime.datetime(int(year), int(month), int(day))

def get_annotation_path(key):
    splitted = key.split("_")
    observation = "_".join(splitted[:4])
    image = "_".join(splitted[:5])
    path = f"labels/{observation}/annotations/{image}_annotations.geojson"
    return path

def get_annotations(gcp_client):
    # Returns dictionary where key is site (e.g. "Ken_Banda_80-35_25-68")
    # and value is list of paths to .annotations.geojson
    all_annotations = list_files_in_bucket_with_suffix(gcp_client, "_annotations.geojson")
    annotations_dict = {}
    for annotation in all_annotations:
        site = annotation.split("/")[1]
        if site in annotations_dict:
            annotations_dict[site].append(annotation)
        else:
            annotations_dict[site] = [annotation]
    return annotations_dict

def annotations_path_to_bs(path:str):
    return path.replace("annotations", "bs").replace(".geojson", ".tif")

def annotations_path_to_rgb(path:str):
    return path.replace("annotations", "rgb").replace(".geojson", ".tif")

def observation_factory(gcp_client, validation_sites: List[str]) -> List[ObservationPointer]:
    # TODO: Pass a function that does train/val split
    for site, annotations in get_annotations(gcp_client).items():
        split = Split.VAL if any(val_site in site for val_site in validation_sites) else Split.TRAIN
        for annotation_path in annotations:
            bs_path = annotations_path_to_bs(annotation_path)
            rgb_path = annotations_path_to_rgb(annotation_path)
            observation = ObservationPointer(
                uri_to_bs=get_public_url(bs_path),
                uri_to_rgb=get_public_url(rgb_path),
                uri_to_annotations=get_public_url(annotation_path),
                split=split
            )
            yield observation
