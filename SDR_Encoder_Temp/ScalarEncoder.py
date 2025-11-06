from typing import List
from SDR_Encoder_Temp.BaseEncoder import BaseEncoder
from SDR import SDR
from dataclasses import dataclass

@dataclass
class ScalarEncoderParameters:
    minimum: float
    maximum: float
    clipInput: bool
    periodic: bool
    category: bool
    activeBits: float
    sparsity: float
    memberSize: int
    radius: float
    resolution: float

#struct ScalarEncoderParameters

class ScalarEncoder(BaseEncoder):

    def __init__(self, parameters: ScalarEncoderParameters, dimensions: List[int]):
        super().__init__(dimensions)
        self.minimum = parameters.minimum
        self.maximum = parameters.maximum
        self.clipInput = parameters.clipInput
        self.periodic = parameters.periodic
        self.category = parameters.category
        self.activeBits = parameters.activeBits
        self.sparsity = parameters.sparsity
        self.memberSize = parameters.memberSize
        self.radius = parameters.radius
        self.resolution = parameters.resolution

    def encode(self, input_value: float, output: SDR) -> None:
        pass

params = ScalarEncoderParameters(
    minimum = 0,
    maximum = 100,
    clipInput = False,
    periodic = False,
    category = False,
    activeBits = 21,
    sparsity = 0,
    memberSize = 500,
    radius = 0,
    resolution = 0,
)
encoder = ScalarEncoder(params ,dimensions=[100,200])
print(encoder.dimensions)
print(encoder.size)