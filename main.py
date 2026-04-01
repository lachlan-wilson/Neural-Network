import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./style.mlpstyle")


class MultilayerPerceptron:

    def __init__(self, sizes):
        self.sizes = sizes

        random_gen = np.random.default_rng(42)
        self.perceptrons = [random_gen.random(i, dtype=np.float32) for i in sizes]
        self.paths = [random_gen.random((sizes[i] * sizes[i + 1], 2), dtype=np.float32) for i in range(len(sizes) - 1)]

    def show_background(self, ax, max_height=None, layer_width=6):
        x_offset = layer_width / 2

        ax.add_patch(plt.Rectangle((0, 0), layer_width, max_height + 4.2, color=(0, 0, 0, 0.05)))
        ax.text(x_offset, max_height + 3, "Input Layer", va="center", ha="center")
        ax.add_patch(plt.Rectangle((layer_width, 0), layer_width * (len(self.sizes) - 2), max_height + 4.2,
                                   color=(0, 0, 0, 0.08)))
        ax.text(layer_width * (len(self.sizes)) / 2, max_height + 3,
                "Hidden Layer" + ("s" if (len(self.sizes) - 2) > 1 else ""), va="center", ha="center")
        ax.add_patch(
            plt.Rectangle((layer_width + layer_width * (len(self.sizes) - 2), 0), layer_width, max_height + 4.2,
                          color=(0, 0, 0, 0.05)))
        ax.text(layer_width * (len(self.sizes) - 0.5), max_height + 3, "Output Layer", va="center", ha="center")

    def show_perceptrons(self, ax, max_height=None, layer_width=6, base_colour=(0, 0, 0)):
        x_offset = layer_width / 2

        for x, layer in enumerate(self.sizes):
            if layer > max_height:

                ax.text(x * layer_width + x_offset, max_height + 1.4, f"{layer}", va="center", ha="center")

                for y in range(max_height)[:int(max_height / 2) - 1]:
                    ax.add_patch(plt.Circle((x * layer_width + x_offset, y + 1),
                                            0.4,
                                            edgecolor=base_colour,
                                            facecolor=(base_colour, self.perceptrons[x][y])))

                for i in [-0.4, 0, 0.4]:
                    ax.add_patch(plt.Circle((x * layer_width + x_offset, (max_height + 1) / 2 + i),
                                            0.08,
                                            color="black"))

                for y in range(max_height)[int(max_height / 2) + 1 + (max_height & 1):]:
                    ax.add_patch(plt.Circle((x * layer_width + x_offset, y + 1),
                                            0.4,
                                            edgecolor=base_colour,
                                            facecolor=(base_colour, self.perceptrons[x][y])))

            else:

                ax.text(x * layer_width + x_offset, max_height + 1.4, f"{layer}", va="center", ha="center")

                for y in range(layer):
                    ax.add_patch(plt.Circle((x * layer_width + x_offset, y + 1 + (max_height - layer) / 2),
                                            0.4,
                                            edgecolor=base_colour,
                                            facecolor=(base_colour, self.perceptrons[x][y])))

    def display(self, max_height=None, layer_width=6, base_colour=(0, 0, 0)):
        if max_height is None:
            max_height = max(self.sizes)

        fig, ax = plt.subplots()

        self.show_background(ax, max_height, layer_width)
        self.show_perceptrons(ax, max_height, layer_width, base_colour)

        # Set limits so circles are visible
        ax.set_xlim(0, len(self.sizes) * layer_width)
        ax.set_ylim(0, max_height + 4.2)

        # Ensure circles stay circular
        ax.set_aspect("equal")

        # Hide axes for a cleaner diagram look
        ax.axis("off")

        plt.title("Multilayer Perceptron", size=20)
        plt.show()


MLP = MultilayerPerceptron([784, 16, 16, 16, 10])
MLP.display(max_height=16, base_colour=(0.1, 0.4, 0.9))
