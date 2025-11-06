from typing import List

class SDR:
    def __init__(self, dimensions: list[int]):
        self.size = 1
        self.dimensions_ = dimensions
        for dim in dimensions:
            self.size *= dim
        self.sparse = []

    def zero(self):
        self.sparse = []

    def getSparse(self):
        return self.sparse

    def setSparse(self, dimensions: List[int]):
        self.sparse = dimensions