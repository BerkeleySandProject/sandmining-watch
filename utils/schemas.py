from enum import Enum
from dataclasses import dataclass

@dataclass
class ObservationPointer:
    uri_to_s1: str
    uri_to_s2: str # This points to a S2 L2A .tif
    uri_to_s2_l1c: str
    uri_to_rgb: str
    uri_to_annotations: str
    uri_to_rivers: str
    name: str
