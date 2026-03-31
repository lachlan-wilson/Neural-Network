import numpy as np
import matplotlib.pyplot as plt


class MultilayerPerceptron:

    def __init__(self, sizes):
        self.sizes = sizes

        random_gen = np.random.default_rng(42)
        self.perceptrons = [random_gen.random(i, dtype=np.float32) for i in sizes]
        self.paths = [random_gen.random((sizes[i] * sizes[i + 1], 2), dtype=np.float32) for i in range(len(sizes) - 1)]

    def display(self):
        fig, ax = plt.subplots()

        for x, layer in enumerate(self.sizes):
            for y in range(layer):
                ax.add_patch(plt.Circle((x * 5 + 1, y + 1), 0.4, color='black', alpha=self.perceptrons[x][y]))

        # Set limits so circles are visible
        ax.set_xlim(0, len(self.sizes) * 5 - 2.6)
        ax.set_ylim(0, max(self.sizes) + 1)

        # Ensure circles stay circular
        ax.set_aspect('equal')

        # Hide axes for a cleaner diagram look
        ax.axis('on')

        plt.title("Multilayer Perceptron")
        plt.show()


MLP = MultilayerPerceptron([16, 16, 16, 16, 16])
MLP.display()