import numpy as np


class MultilayerPerceptron:

    def __init__(self, sizes):
        random_gen = np.random.default_rng()
        self.perceptrons = [np.random(i, dtype=np.float32) for i in sizes]
        self.paths = [np.random((sizes[i] * sizes[i + 1], 2), dtype=np.float32) for i in range(len(sizes) - 1)]


MLP = MultilayerPerceptron([784, 16, 16, 16, 10])
print(MLP.perceptrons)
print(MLP.paths)