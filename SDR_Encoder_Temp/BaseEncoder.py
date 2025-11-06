from abc import ABC, abstractmethod


class BaseEncoder(ABC):

    def __init__(self, dimensions=None):
        self._dimensions = None
        self.size = None

        if dimensions is not None:
            self.initialize(dimensions)
     #
     # Members dimensions & size describe the shape of the encoded output SDR.
     # This is the total number of bits in the result.
     #
    @property
    def dimensions(self):
        return self._dimensions

    @property
    def size(self):
        return self.size

    def initialize(self, dimensions):
        self._dimensions = list(dimensions)
        #self.size = SDR(dimensions).size

    def reset(self):
        pass

    @abstractmethod
    def encode(self, input_value, output):
        raise NotImplementedError("Subclasses must implement this method")