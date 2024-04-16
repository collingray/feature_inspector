from dataclasses import dataclass
from typing import List


@dataclass
class FeatureExample:
    activation: float
    context: str


@dataclass
class Feature:
    num: int
    layers: List[List[FeatureExample]]
