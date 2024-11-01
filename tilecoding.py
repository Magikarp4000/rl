import numpy as np


class TileCoding:
    def __init__(self, n: int, dim: int, bounds: list, num_per_dim: list, offsets: list, mod_list=[]):
        assert(len(bounds) == dim)
        assert(len(num_per_dim) == dim)
        assert(len(offsets) == dim)

        for cur_bounds in bounds:
            if isinstance(cur_bounds, int):
                cur_bounds = [cur_bounds]
            assert(len(cur_bounds) == 2)
        
        self.n = n
        self.dim = dim
        self.bounds = bounds
        self.num_per_dim = num_per_dim
        self.offsets = offsets
        self.mod_list = mod_list
        self.total = int(np.prod(np.array(self.num_per_dim) + 1))

    def encode(self, state: list):
        assert(len(state) == self.dim)

        res = []
        for i in range(self.n):
            encoding = i * self.total
            multiplier = 1

            for j in range(self.dim):
                cur_bounds = self.bounds[j]

                length = cur_bounds[1] - cur_bounds[0]
                size = length / self.num_per_dim[j]

                offset = (i * self.offsets[j] * (size / self.n)) % size
                cur_raw = (state[j] + offset - cur_bounds[0]) / size
                cur = int(np.floor(cur_raw))
                if self.mod_list and self.mod_list[j]:
                    cur %= self.n + 1

                encoding += cur * multiplier
                multiplier *= self.num_per_dim[j] + 1
            res.append(encoding)
        return res

    def decode(self, encoding: list):
        decoding = []
        for code in encoding:

            mod_code = code % self.total
            cur_decoding = []

            for num in self.num_per_dim:
                mod_code, value = divmod(mod_code, num)
                cur_decoding.append(value)
            
            decoding.append(cur_decoding)
        return decoding
