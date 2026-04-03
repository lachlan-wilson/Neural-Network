# ----- Multilayer Perceptron -----
#       ----- 31/03/2026 -----

# Standard libraries
from os.path import join

# External libraries
import matplotlib.pyplot as plt
import numpy as np

# Internal libraries
import mnist_reader


def load_dataset(input_path):
    """
    Read the MNIST dataset into training and testing data.

    Parameters
    ----------
    input_path: str
        The directory containing the MNIST .ubyte files.

    Returns
    -------
    tuple
        A nested tuple in the format ((x_train, y_train), (x_test, y_test)).

    Notes
    -----
    .. [1] https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
    """
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    plt.style.use("./style.mlpstyle")
    mnist_dataloader = mnist_reader.MnistDataloader(training_images_filepath, training_labels_filepath,
                                                    test_images_filepath,
                                                    test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    return (x_train, y_train), (x_test, y_test)


def non_linear(x):
    """
    Make most values in the middle but extreme ones more extreme.
    Follows function f(x) = ½(64(x-0.5)^7 + 7x + 0.5)

    Parameters
    ----------
    x: float
        A value between 0 and 1 inclusive.

    Returns
    -------
    float
        A value between 0 and 1 inclusive.
    """
    assert 0 <= x <= 1, "x must be between 0 and 1 inclusive."
    return (64 * (x - 0.5) ** 7 + x + 0.5) / 2


def sigmoid(x):
    """
    Make any value a value between 0 and 1 exclusive.
    Follows function f(x) = 1/(1+e^-x)

    Parameters
    ----------
    x: float
        TODO make a numpy array

    Returns
    -------
    float
        A value between 0 and 1 inclusive.
    """
    return 1 / (1 + np.e ** -x)


class Diagram(plt.Axes):
    def __init__(self, mlp, figure, max_height=None, layer_width=6, output_labels=None, ):
        # Ensure all the parameters are valid
        assert isinstance(mlp, MultilayerPerceptron), "mlp must be a MultilayerPerceptron."
        assert isinstance(figure, plt.Figure), "figure must be a Figure."
        assert isinstance(max_height, (int, float)) and max_height > 0, "max_height must be a positive, real number."
        assert isinstance(layer_width, (int, float)) and layer_width > 0, "layer_width must be a positive, real number."
        assert output_labels is None or (isinstance(output_labels, list) and all(
            isinstance(label, str) for label in output_labels)
                                         ), "output_labels must be None or list of strings."

        plt.style.use("./style.mlpstyle")  # Use the styles located at ./styles.mlpstyle

        # Store variables as class attributes
        self.figure = figure
        self.mlp = mlp
        self.layer_width = layer_width
        self.x_offset = layer_width / 2
        self.output_labels = output_labels

        # If max height wasn't defined set it to the biggest layer size
        if max_height is None or max_height > max(self.mlp.sizes):
            self.max_height = max(self.mlp.sizes)
        else:
            self.max_height = max_height

    def add_box(self, start, width, title):
        """
        Add a box to the plot at a given position.

        Parameters
        ----------
        start: tuple of floats
            Coordinate (x, y) for bottom left of box.
        width: float
            Width of the box.
        title: str
            Title of the box, displayed at the centre of the top.
        """
        # Add the box
        self.add_patch(plt.Rectangle(start,
                                     width,
                                     self.max_height + 4.2,
                                     color=(0, 0, 0, 0.05)
                                     ))
        # Add the title
        self.text(self.x_offset,
                  self.max_height + 3,
                  title,
                  va="center",  # Centre the title on its coordinates
                  ha="center"
                  )

    def add_circle(self, centre, colour=(1, 1, 1, 1)):
        """
        Add a circle to the plot at a given position.

        Parameters
        ----------
        centre: tuple of floats
            Coordinate (x, y) of the centre of the circle.
        colour: tuple of floats
            RBGA value of the colour of the circle.
        """
        # Add the circle
        self.add_patch(plt.Circle(centre,
                                  0.4,
                                  edgecolor=(colour[:-1], 1),
                                  facecolor=colour,
                                  zorder=2  # Add circles above lines
                                  ))

    def add_line(self, xs, ys, colour):
        """
        Add a line to the plot at given positions.

        Parameters
        ----------
        xs: list of floats
            List of the x-ordinates [x1, x2].
        ys: list of floats
            List of the y-ordinates [y1, y2].
        colour: tuple of floats
            RGBA value of the colour of the line.
        """
        self.plot(xs,
                  ys,
                  color=colour,
                  linewidth=colour[-1]
                  )

    def add_layer_title(self, x):
        """
        Add a title for a layer to the plot at a given layer.

        Parameters
        ----------
        x: int
            The x-ordinate of the centre of the layer.
        """
        self.text(x * self.layer_width + self.x_offset,
                  self.max_height + 1.4,
                  f"{len(self.mlp.activations[x])}",
                  va="center",
                  ha="center"
                  )

    def add_perceptron(self, x, y, reverse=False):
        """
        Add a perceptron to the plot at a given position.

        Parameters
        ----------
        x: int
            The x-ordinate of the layer.
        y: int
            The y-ordinate of the perceptron.
        reverse: bool
            If it is a perceptron from the bottom when perceptrons are being skipped.
        """
        # Add a solid background circle
        self.add_circle((x * self.layer_width + self.x_offset, y + 1))
        # Add a circle in the same position where the colour represents the activation of the perceptron and opacity the bias
        self.add_circle((x * self.layer_width + self.x_offset, y + 1),
                        (1 - self.mlp.activations[x][y if not reverse else -y],
                         self.mlp.activations[x][y if not reverse else -y],
                         0,
                         1 if x == 0 else non_linear(sigmoid(self.mlp.biases[x - 1][y if not reverse else -y]))
                         ))

        # If it's not the output layer
        if x < len(self.mlp.sizes):
            # Loop for each perceptron in the next layer
            for y2 in range(self.mlp.sizes[x + 1]):
                # Draw a line between the centre of the two perceptrons
                self.add_line([x * self.layer_width + self.x_offset,
                               (x + 1) * self.layer_width + self.x_offset],
                              [y + 1, 1 + (y2 if self.mlp.sizes[x + 1] >= self.max_height else
                                           y2 + (self.max_height - self.mlp.sizes[x + 1])) / 2],
                              (1 - self.mlp.activations[x][y],
                               self.mlp.activations[x][y],
                               0,
                               non_linear(sigmoid(self.mlp.weights[x][y][1 + (
                                   y2 if self.mlp.sizes[x + 1] >= self.max_height else
                                   y2 + (self.max_height - self.mlp.sizes[x + 1]) / 2
                               )]))))

    def show(self):
        """
        Display the plot.
        """
        # Add the background boxes
        self.add_box((0, 0), self.layer_width, "Input Layer")
        self.add_box((self.layer_width, 0), self.layer_width * (len(self.mlp.sizes) - 2), "Hidden Layer")
        self.add_box((self.layer_width * (len(self.mlp.sizes) - 1), 0), self.layer_width, "Output Layer")

        # Loop for each layer
        for x in range(len(self.mlp.sizes)):
            # If the layer is too tall
            if self.mlp.sizes[x] > self.max_height:

                self.add_layer_title(x)

                # Add the top perceptrons
                for y in range(self.max_height)[:int(self.max_height / 2) - 1]:
                    self.add_perceptron(x, y, True)

                # Add an ellipsis in the middle
                for i in [-0.4, 0, 0.4]:
                    self.add_patch(plt.Circle((x * self.layer_width + self.x_offset, (self.max_height + 1) / 2 + i),
                                              0.08,
                                              color="black"
                                              ))

                # Add the bottom perceptrons
                for y in range(self.max_height)[int(self.max_height / 2) + 1 + (self.max_height & 1):]:
                    self.add_perceptron(x, y)

            # Otherwise
            else:
                self.add_layer_title(x)

                # Add the perceptrons in the vertical centre
                for y in range(self.mlp.sizes[x]):
                    self.add_perceptron(x, int(y + (self.max_height - len(self.mlp.activations[x + 1])) / 2 + 1))

        if self.output_labels is not None:
            for y, label in enumerate(self.output_labels):
                self.text(len(self.mlp.sizes) * self.layer_width - self.x_offset + 1,
                          y + (self.max_height - self.mlp.sizes[-1]) / 2 + 1,
                          label,
                          ha="center",
                          va="center"
                          )
        self.show()
        self.set_xlim(0, len(self.sizes) * self.layer_width)
        self.set_ylim(0, self.max_height + 4.2)
        self.text(len(self.sizes) * self.layer_width / 2,
                  -1,
                  "Colour: Activation  (red: low, green: high)    Opacity: Weight/Bias",
                  ha="center"
                  )
        self.axis("off")
        self.set_aspect("equal")
        self.title("Multilayer Perceptron", size=20)


class MultilayerPerceptron:
    """
    The class for the multilayer perceptron.

    Methods
    -------

    """

    # Initialise the class with activations, biases and weights based on the entered size array
    def __init__(self, sizes):
        self.sizes = sizes

        random_gen = np.random.default_rng(42)
        self.activations = [random_gen.random(i, dtype=np.float32) for i in sizes]
        self.biases = [10 * random_gen.random(i, dtype=np.float32) - 5 for i in sizes[1:]]
        self.weights = [10 * random_gen.random((sizes[i], sizes[i + 1])) - 5 for i in range(len(sizes) - 1)]

    def show_background(self, ax, max_height, layer_width):
        x_offset = layer_width / 2

        ax.add_patch(plt.Rectangle((0, 0),
                                   layer_width,
                                   max_height + 4.2,
                                   color=(0, 0, 0, 0.05)
                                   ))
        ax.text(x_offset,
                max_height + 3,
                "Input Layer",
                va="center",
                ha="center"
                )
        ax.add_patch(plt.Rectangle((layer_width, 0),
                                   layer_width * (len(self.sizes) - 2),
                                   max_height + 4.2,
                                   color=(0, 0, 0, 0.08)
                                   ))
        ax.text(layer_width * (len(self.sizes)) / 2,
                max_height + 3,
                "Hidden Layer" + ("s" if (len(self.sizes) - 2) > 1 else ""),
                va="center",
                ha="center"
                )
        ax.add_patch(plt.Rectangle((layer_width + layer_width * (len(self.sizes) - 2), 0),
                                   layer_width,
                                   max_height + 4.2,
                                   color=(0, 0, 0, 0.05)
                                   ))
        ax.text(layer_width * (len(self.sizes) - 0.5),
                max_height + 3,
                "Output Layer",
                va="center",
                ha="center"
                )

    def show_perceptrons(self, ax, max_height, layer_width):

        def draw_circle(colour):
            ax.add_patch(plt.Circle((x * layer_width + x_offset,
                                     y + 1 if self.sizes[x] >= max_height else y + (
                                             max_height - self.sizes[x]) / 2 + 1),
                                    0.4,
                                    edgecolor=(colour[:-1], 1),
                                    facecolor=colour,
                                    zorder=10
                                    ))

        def show_perceptron(inverse=False):
            try:
                for y2 in range(max_height if self.sizes[x + 1] > max_height else self.sizes[x + 1]):
                    ax.plot([x * layer_width + x_offset,
                             (x + 1) * layer_width + x_offset],
                            [y + 1 if self.sizes[x] >= max_height else y + (
                                    max_height - self.sizes[x]) / 2 + 1,
                             y2 + 1 if self.sizes[x + 1] >= max_height else y2 + (
                                     max_height - self.sizes[x + 1]) / 2 + 1],
                            color=(1 - self.activations[x][y], self.activations[x][y], 0,
                                   non_linear(sigmoid(self.weights[x][y][y2]))),
                            linewidth=non_linear(sigmoid(self.weights[x][y][y2]))
                            )
            except IndexError:
                pass

            draw_circle((1, 1, 1, 1))
            draw_circle((1 - self.activations[x][y if not inverse else -y],
                         self.activations[x][y if not inverse else -y],
                         0,
                         1 if x == 0 else non_linear(sigmoid(self.biases[x - 1][y if not inverse else -y]))
                         ))

        x_offset = layer_width / 2

        for x in range(len(self.sizes)):
            if self.sizes[x] > max_height:

                ax.text(x * layer_width + x_offset,
                        max_height + 1.4,
                        f"{self.sizes[x]}",
                        va="center",
                        ha="center"
                        )

                for y in range(max_height)[:int(max_height / 2) - 1]:
                    show_perceptron(inverse=True)

                for i in [-0.4, 0, 0.4]:
                    ax.add_patch(plt.Circle((x * layer_width + x_offset, (max_height + 1) / 2 + i),
                                            0.08,
                                            color="black"
                                            ))

                for y in range(max_height)[int(max_height / 2) + 1 + (max_height & 1):]:
                    show_perceptron()
            else:

                ax.text(x * layer_width + x_offset,
                        max_height + 1.4,
                        f"{self.sizes[x]}",
                        va="center",
                        ha="center"
                        )

                for y in range(self.sizes[x]):
                    show_perceptron()

    def display(self, max_height=None, layer_width=6, output_labels=None):
        if max_height is None or max_height > max(self.sizes):
            max_height = max(self.sizes)

        plt.style.use("./style.mlpstyle")
        fig, ax = plt.subplots()

        self.show_background(ax, max_height, layer_width)
        self.show_perceptrons(ax, max_height, layer_width)

        for y, label in enumerate(output_labels):
            ax.text((len(self.sizes) - 0.5) * layer_width + 1,
                    y + 1 + (max_height - self.sizes[-1]) / 2,
                    label,
                    ha="center",
                    va="center"
                    )

        # Set limits so circles are visible
        ax.set_xlim(0, len(self.sizes) * layer_width)
        ax.set_ylim(0, max_height + 4.2)
        ax.text(len(self.sizes) * layer_width / 2,
                -1,
                "Colour: Activation  (red: low, green: high)    Opacity: Weight/Bias",
                ha="center"
                )

        # Ensure circles stay circular
        ax.set_aspect("equal")

        # Hide axes for a cleaner diagram look
        ax.axis("off")

        plt.title("Multilayer Perceptron", size=20)
        plt.show()

    def display_display(self, max_height=None, layer_width=6, output_labels=None):
        assert isinstance(max_height, (int, float)) and max_height > 0, "max_height must be a positive, real number."
        assert isinstance(layer_width, (int, float)) and layer_width > 0, "layer_width must be a positive, real number."
        assert output_labels is None or (isinstance(output_labels, list) and all(
            isinstance(label, str) for label in output_labels)), "output_labels must be None or list of strings."

        fig = plt.figure()
        diagram = Diagram(self, fig, max_height, layer_width, output_labels)
        diagram.show()

    def calculate_activations(self):
        for x in range(1, len(self.sizes)):
            self.activations[x] = sigmoid(self.activations[x - 1] @ self.weights[x - 1] + self.biases[x - 1])


MLP = MultilayerPerceptron([748, 16, 16, 16, 10])
MLP.calculate_activations()
MLP.display_display(max_height=16, output_labels=[str(i) for i in range(1, 11)])
# MLP.display(max_height=16, output_labels=[str(i) for i in range(1, 11)])

(training_images, training_outputs), (testing_images, testing_outputs) = load_dataset("./mnist-dataset")
