from enum import Enum
from dataclasses import dataclass

class Split(Enum):
    TRAIN = "train"
    VAL = "val"

@dataclass
class ObservationPointer:
    uri_to_bs: str
    uri_to_rgb: str
    uri_to_annotations: str
    split: Split

