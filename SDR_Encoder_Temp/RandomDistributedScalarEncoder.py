import math
import struct
import mmh3

from SDR_Encoder_Temp.BaseEncoder import BaseEncoder
from dataclasses import dataclass
from typing import List
from SDR import SDR

@dataclass
class RDSEParameters:
    size: int
    activeBits: int
    sparsity: float
    radius: float
    resolution: int
    category: bool
    seed: int

class RandomDistributedScalarEncoder(BaseEncoder):
    def __init__(self, parameters: RDSEParameters, dimensions: List[int]):
        super().__init__(dimensions)
        self.memberSize = parameters.size
        self.activeBits = parameters.activeBits
        self.sparsity = parameters.sparsity
        self.radius = parameters.radius
        self.resolution = parameters.resolution
        self.category = parameters.category
        self.seed = parameters.seed

    def encode(self, input_value: float, output: SDR) -> None:
        assert output.size == self.size, "Output SDR size does not match encoder size."
        if math.isnan(input_value):
            output.zero()
            return
        if self.category:
            if input_value != int(input_value) or input_value < 0:
                raise ValueError("Input to category encoder must be an unsigned integer")

        data = [0] * self.size

        index = int(input_value / self.resolution)

        for offset in range(self.activeBits):
            hash_buffer = index + offset
            bucket = mmh3.hash(struct.pack('I', hash_buffer), self.seed, signed=False)
            bucket = bucket % self.size
            data[bucket] = 1

        output.setSparse(data) #we may need setDense implemented for SDR class



#After encode we may need a check_parameters method since most of the encoders have this


#Tests
params = RDSEParameters(
    size = 500,
    activeBits = 21,
    sparsity = 0,
    radius = 0,
    resolution = 0,
    category = False,
    seed = 0
)
encoder = RandomDistributedScalarEncoder(params, dimensions=[params.size])
output = SDR(dimensions=[params.size])
encoder.encode(7.3, output)
print(output.getSparse())