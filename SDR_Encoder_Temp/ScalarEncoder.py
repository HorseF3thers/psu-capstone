from typing import List
from SDR_Encoder_Temp.BaseEncoder import BaseEncoder
from SDR import SDR
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class ScalarEncoderParameters:
    minimum: float
    maximum: float
    clipInput: bool
    periodic: bool
    category: bool
    activeBits: int
    sparsity: float
    memberSize: int
    radius: float
    resolution: float

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
        assert output.size == self.size, "Output SDR size does not match encoder size."

        if math.isnan(input_value):
            output.zero()
            return

        if self.clipInput:
            if self.periodic:
                raise NotImplementedError("Periodic input clipping not implemented.")
            else:
                input_value = max(input_value, self.minimum)
                input_value = min(input_value, self.maximum)
        else:
            if self.category:
                if input_value != float(int(input_value)):
                    raise ValueError("Input to category encoder must be an unsigned integer!")
            if not (self.minimum <= input_value <= self.maximum):
                raise ValueError(
                    f"Input must be within range [{self.minimum}, {self.maximum}]! "
                    f"Received {input_value}"
                )

        scale = self.maximum - self.minimum
        result = (input_value - self.minimum) / scale * (self.memberSize - self.activeBits)
        start = int(np.clip(np.round(result), 0, self.memberSize - self.activeBits))

        if not self.periodic:
            start = min(start, output.size - self.activeBits)

        sparse = list(range(start, start + self.activeBits))

        if self.periodic:
            sparse = [bit % output.size for bit in sparse]
            sparse.sort()

        output.setSparse(sparse)

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
    resolution = 0
)
'''encoder = ScalarEncoder(params ,dimensions=[100])
sdr = SDR(dimensions=[10, 10])
print(sdr.size)
sdr.setSparse([0,5,22,99])
print(sdr.getSparse())
sdr.zero()
print(sdr.getSparse())

encoder2 = ScalarEncoder(params ,dimensions=[100])
print(encoder2.size)
print(encoder2.dimensions)'''

encoder3 = ScalarEncoder(params ,dimensions=[params.memberSize])
output = SDR(dimensions=[params.memberSize])
encoder3.encode(7.3, output)
print(output.getSparse())