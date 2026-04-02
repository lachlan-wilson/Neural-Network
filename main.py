import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./style.mlpstyle")


def non_linear(x):
    return (64 * (x - 0.5) ** 7 + x + 0.5) / 2


class MultilayerPerceptron:

    def __init__(self, sizes):
        self.sizes = sizes

        random_gen = np.random.default_rng(42)
        self.perceptrons = [random_gen.random(i, dtype=np.float32) for i in sizes]
        self.biases = [random_gen.random(i, dtype=np.float32) for i in sizes[1:]]
        self.weights = [random_gen.random((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]

    def show_background(self, ax, max_height, layer_width):
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

    def show_perceptrons(self, ax, max_height, layer_width):

        def draw_circle(colour):
            ax.add_patch(plt.Circle((x * layer_width + x_offset,
                                     y + 1 if self.sizes[x] >= max_height else y + (
                                             max_height - self.sizes[x]) / 2 + 1),
                                    0.4,
                                    edgecolor=(0, 0, 0),
                                    facecolor=colour,
                                    zorder=10))

        def show_perceptron():
            try:
                for y2 in range(max_height if self.sizes[x + 1] > max_height else self.sizes[x + 1]):
                    ax.plot([x * layer_width + x_offset,
                             (x + 1) * layer_width + x_offset],
                            [y + 1 if self.sizes[x] >= max_height else y + (
                                    max_height - self.sizes[x]) / 2 + 1,
                             y2 + 1 if self.sizes[x + 1] >= max_height else y2 + (
                                     max_height - self.sizes[x + 1]) / 2 + 1],
                            color=(1 - self.perceptrons[x][y], self.perceptrons[x][y], 0,
                                   non_linear(self.weights[x][y][y2])),
                            linewidth=1
                            )
            except IndexError:
                pass

            draw_circle((1, 1, 1))
            draw_circle((1 - self.perceptrons[x][y],
                         self.perceptrons[x][y],
                         0,
                         1 if x == 0 else non_linear(self.biases[x - 1][y])))

        x_offset = layer_width / 2

        for x in range(len(self.sizes)):
            if self.sizes[x] > max_height:

                ax.text(x * layer_width + x_offset, max_height + 1.4, f"{self.sizes[x]}", va="center", ha="center")

                for y in range(max_height)[:int(max_height / 2) - 1]:
                    show_perceptron()

                for i in [-0.4, 0, 0.4]:
                    ax.add_patch(plt.Circle((x * layer_width + x_offset, (max_height + 1) / 2 + i),
                                            0.08,
                                            color="black"))

                for y in range(max_height)[int(max_height / 2) + 1 + (max_height & 1):]:
                    show_perceptron()
            else:

                ax.text(x * layer_width + x_offset, max_height + 1.4, f"{self.sizes[x]}", va="center", ha="center")

                for y in range(self.sizes[x]):
                    show_perceptron()

    def display(self, max_height=None, layer_width=6, output_labels=None):
        if max_height is None:
            max_height = max(self.sizes)

        fig, ax = plt.subplots()

        self.show_background(ax, max_height, layer_width)
        self.show_perceptrons(ax, max_height, layer_width)

        for y, label in enumerate(output_labels):
            ax.text((len(self.sizes) - 0.5) * layer_width + 1,
                    y + 1 + (max_height - self.sizes[-1]) / 2,
                    str(label),
                    ha="center",
                    va="center")

        # Set limits so circles are visible
        ax.set_xlim(0, len(self.sizes) * layer_width)
        ax.set_ylim(0, max_height + 4.2)
        ax.text(len(self.sizes) * layer_width / 2,
                -1,
                "Activation: Colour (low: red high: green)    Weight/Bias: Opacity",
                ha="center")

        # Ensure circles stay circular
        ax.set_aspect("equal")

        # Hide axes for a cleaner diagram look
        ax.axis("off")

        plt.title("Multilayer Perceptron", size=20)
        plt.show()


MLP = MultilayerPerceptron([784, 16, 16, 16, 10])
MLP.display(max_height=16, output_labels=range(1, 11))
