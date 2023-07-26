from enum import Enum
from dataclasses import dataclass

@dataclass
class ObservationPointer:
    uri_to_bs: str
    uri_to_rgb: str
    uri_to_annotations: str
    name: str
