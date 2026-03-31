import matplotlib.pyplot as plt
import numpy as np


class MultilayerPerceptron:

    def __init__(self, sizes):
        self.sizes = sizes

        random_gen = np.random.default_rng(42)
        plt.rcParams['text.usetex'] = True

        self.perceptrons = [random_gen.random(i, dtype=np.float32) for i in sizes]
        self.paths = [random_gen.random((sizes[i] * sizes[i + 1], 2), dtype=np.float32) for i in range(len(sizes) - 1)]

    def display(self, max_height=None):

        if max_height is None:
            max_height = max(self.sizes)

        fig, ax = plt.subplots()
        x_offset = 1
        for x, layer in enumerate(self.sizes):
            if layer > max_height:
                x_offset = 4
                for y in range(max_height)[:int(max_height/2) - 1]:
                    ax.add_patch(plt.Circle((x * 5 + x_offset, y + 1),
                                            0.4,
                                            color='red',
                                            alpha=self.perceptrons[x][y]))

                for i in [-0.4, 0, 0.4]:
                    ax.add_patch(plt.Circle((x * 5 + x_offset, (max_height+1)/2 + i),
                                            0.08,
                                            color='black'))

                ax.text(x * 5 + x_offset - 1, (max_height+1)/2, r"|\nabla\phi| &=& 1,\\")

                for y in range(max_height)[int(max_height/2) + 1 + (max_height & 1):]:
                    ax.add_patch(plt.Circle((x * 5 + x_offset, y + 1),
                                            0.4,
                                            color='red',
                                            alpha=self.perceptrons[x][y]))

            else:
                for y in range(layer):
                    ax.add_patch(plt.Circle((x * 5 + x_offset, y + 1 + (max_height - layer) / 2),
                                            0.4,
                                            color='red',
                                            alpha=self.perceptrons[x][y]))

        # Set limits so circles are visible
        ax.set_xlim(0, len(self.sizes) * 5 - 3.6 + x_offset)
        ax.set_ylim(0, max_height + 1)

        # Ensure circles stay circular
        ax.set_aspect('equal')

        # Hide axes for a cleaner diagram look
        ax.axis('on')

        plt.title("Multilayer Perceptron")
        plt.show()


MLP = MultilayerPerceptron([784, 16, 16, 16, 10])
MLP.display(16)
