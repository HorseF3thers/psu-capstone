class SDR:
    def __init__(self, dimensions):
        self.size = 1
        self.dimensions_ = dimensions
        for dim in dimensions:
            self.size *= dim

