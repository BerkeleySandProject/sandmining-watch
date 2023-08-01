from enum import Enum
from dataclasses import dataclass

@dataclass
class ObservationPointer:
    uri_to_s1: str
    uri_to_s2: str
    uri_to_rgb: str
    uri_to_annotations: str
    name: str
