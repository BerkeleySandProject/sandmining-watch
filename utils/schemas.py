from enum import Enum
from dataclasses import dataclass
from typing import Optional

@dataclass
class ObservationPointer:
    uri_to_s1: str
    uri_to_s2: str # This points to a S2 L2A .tif
    uri_to_s2_l1c: str
    uri_to_rgb: str
    uri_to_annotations: Optional[str] # In inference mode, we don't have annotations
    uri_to_rivers: str
    name: str
    cluster_id: Optional[int] # This is the spatial cluster number
    latitude: Optional[float] #latitude of the centroid of the observation
    longitude: Optional[float]
    date: Optional[str]
    # num_polygons: Optional[int] #number of polygons in the annotation
