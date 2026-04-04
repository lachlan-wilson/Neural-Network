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
    x: ndarray(dtype=float, ndim=1)
        A numpy array of the x values

    Returns
    -------
    ndarray(dtype=float, ndim=1)
        A numpy array of values between 0 and 1 inclusive.
    """
    return 1 / (1 + np.e ** -x)


class Diagram(plt.Axes):
    """
    Axes that plots a diagram of the multilayer perceptron based on a `MultiLayerPerceptron` object.

    Attributes
    ----------
    mlp: MultilayerPerceptron
            An instance of the `MultilayerPerceptron` object.
    max_height: int
        The maximum number of perceptrons shown in one layer.
    layer_width: int
        The width of each layer.
    x_offset: float
        The x shift of each perceptron, `layer_width / 2`.
    output_labels: list of str
        A list containing the names of the output perceptrons.

    Methods
    -------
    show_diagram()
        Display the diagram in its `Figure`
    """
    def __init__(self, mlp, __figure, max_height=None, layer_width=6, output_labels=None, **kwargs):
        """
        Initialise the `Diagram` object with the given parameters. Assigns itself to the global figure.

        Parameters
        ----------
        mlp: MultilayerPerceptron
            An instance of the `MultilayerPerceptron` object.
        __figure: Figure
            The `Figure` object the diagram should be contained in.
        max_height: int, default=None
            The maximum number of perceptrons shown in one layer.
        layer_width: int, default=6
            The width of each layer.
        output_labels: list of str, default=
            A list containing the names of the output perceptrons
        **kwargs: dict
            Extra arguments to `Figure`: refer to `Figure` documentation for a
            list of all possible arguments.
        """
        # Ensure all the parameters are valid
        assert isinstance(mlp, MultilayerPerceptron), "mlp must be a MultilayerPerceptron."
        assert isinstance(__figure, plt.Figure), "__figure must be a Figure."
        assert isinstance(max_height, (int, float)) and max_height > 0, "max_height must be a positive, real number."
        assert isinstance(layer_width, (int, float)) and layer_width > 0, "layer_width must be a positive, real number."
        assert output_labels is None or (isinstance(output_labels, list) and all(
            isinstance(label, str) for label in output_labels)
                                         ), "output_labels must be None or list of strings."

        super().__init__(__figure, [0, 0, 1, 1], **kwargs)
        __figure.add_axes(self)
        plt.style.use("./style.mlpstyle")  # Use the styles located at ./styles.mlpstyle

        # Store variables as class attributes
        self.figure = __figure
        self.mlp = mlp
        self.layer_width = layer_width
        self.x_offset = layer_width / 2
        self.output_labels = output_labels

        # If max height wasn't defined set it to the biggest layer size
        if max_height is None or max_height > max(self.mlp.sizes):
            self.max_height = max(self.mlp.sizes)
        else:
            self.max_height = max_height

    def _add_box(self, start, width, title):
        """
        Add a box to the plot at a given position.

        Parameters
        ----------
        start: tuple of float
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
        self.text(start[0] + width / 2,
                  self.max_height + 3,
                  title,
                  va="center",  # Centre the title on its coordinates
                  ha="center"
                  )

    def _add_circle(self, centre, colour=(1, 1, 1, 1)):
        """
        Add a circle to the plot at a given position.

        Parameters
        ----------
        centre: tuple of float
            Coordinate (x, y) of the centre of the circle.
        colour: tuple of float, default=(1, 1, 1, 1) (white)
            RBGA value of the colour of the circle.
        """
        # Add the circle
        self.add_patch(plt.Circle(centre,
                                  0.4,
                                  edgecolor=(colour[:-1], 1),
                                  facecolor=colour,
                                  zorder=2  # Add circles above lines
                                  ))

    def _add_line_(self, xs, ys, colour):
        """
        Add a line to the plot at given positions.

        Parameters
        ----------
        xs: list of float
            List of the x-ordinates [x1, x2].
        ys: list of float
            List of the y-ordinates [y1, y2].
        colour: tuple of float
            RGBA value of the colour of the line.
        """
        self.plot(xs,
                  ys,
                  color=colour,
                  linewidth=colour[-1]
                  )

    def _add_layer_title(self, x):
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

    def _add_perceptron(self, x, y, ypos=None, reverse=False):
        """
        Add a perceptron to the plot at a given position.

        Parameters
        ----------
        x: int
            The x-ordinate of the layer.
        y: int
            The y-ordinate of the perceptron, used as the index.
        ypos: int, default=None
            The y-ordinate of the perceptron, used to place the perceptron.
        reverse: bool, default=False
            If it is a perceptron from the bottom when perceptrons are being skipped.
        """
        if ypos is None:
            ypos = y
        # Add a solid background circle
        self._add_circle((x * self.layer_width + self.x_offset, ypos + 1))
        # Add a circle in the same position where the colour represents the activation of the perceptron and opacity the bias
        self._add_circle((x * self.layer_width + self.x_offset, ypos + 1),
                         (1 - self.mlp.activations[x][y if not reverse else -y],
                         self.mlp.activations[x][y if not reverse else -y],
                         0,
                         1 if x == 0 else non_linear(sigmoid(self.mlp.biases[x - 1][y if not reverse else -y]))
                         ))

        # If it's not the output layer
        if x < len(self.mlp.sizes) - 1:
            # Loop for each perceptron in the next layer
            for y2 in range(self.mlp.sizes[x + 1]):
                # Draw a line between the centre of the two perceptrons
                self._add_line_([x * self.layer_width + self.x_offset,
                                 (x + 1) * self.layer_width + self.x_offset],
                                [ypos + 1, 1 + (y2 if self.mlp.sizes[x + 1] >= self.max_height else
                                                 y2 + (self.max_height - self.mlp.sizes[x + 1]) / 2)],
                                (1 - self.mlp.activations[x][y if not reverse else -y],
                                  self.mlp.activations[x][y if not reverse else -y],
                                  0,
                                  non_linear(sigmoid(self.mlp.weights[x][y if not reverse else -y][
                                                         y2 if not (
                                                                 self.mlp.sizes[x + 1] > self.max_height and
                                                                 y2 < (self.mlp.sizes[x + 1] / 2)
                                                         ) else -y2]))))

    def show_diagram(self):
        """
        Display the plot.
        """
        # Add the background boxes
        self._add_box((0, 0), self.layer_width, "Input Layer")
        self._add_box((self.layer_width, 0), self.layer_width * (len(self.mlp.sizes) - 2), "Hidden Layer")
        self._add_box((self.layer_width * (len(self.mlp.sizes) - 1), 0), self.layer_width, "Output Layer")

        # Loop for each layer
        for x in range(len(self.mlp.sizes)):
            # If the layer is too tall
            if self.mlp.sizes[x] > self.max_height:

                self._add_layer_title(x)

                # Add the top perceptrons
                for y in range(self.max_height)[int(self.max_height / 2) + 1 + (self.max_height & 1):]:
                    self._add_perceptron(x, y)

                # Add an ellipsis in the middle
                for i in [-0.4, 0, 0.4]:
                    self.add_patch(plt.Circle((x * self.layer_width + self.x_offset, (self.max_height + 1) / 2 + i),
                                              0.08,
                                              color="black"
                                              ))

                # Add the bottom perceptrons
                for y in range(self.max_height)[:int(self.max_height / 2) - 1]:
                    self._add_perceptron(x, y, reverse=True)

            # Else if
            elif self.mlp.sizes[x] == self.max_height:
                self._add_layer_title(x)

                for y in range(self.mlp.sizes[x]):
                    self._add_perceptron(x, y)

            else:
                self._add_layer_title(x)

                # Add the perceptrons in the vertical centre
                for y in range(self.mlp.sizes[x]):
                    self._add_perceptron(x, y, ypos=(y + (self.max_height - self.mlp.sizes[x]) / 2))

        if self.output_labels is not None:
            for y, label in enumerate(self.output_labels):
                self.text(len(self.mlp.sizes) * self.layer_width - self.x_offset + 1,
                          y + (self.max_height - self.mlp.sizes[-1]) / 2 + 1,
                          label,
                          ha="center",
                          va="center"
                          )

        self.set_xlim(0, len(self.mlp.sizes) * self.layer_width)
        self.set_ylim(0, self.max_height + 4.2)
        self.text(len(self.mlp.sizes) * self.layer_width / 2,
                  -1,
                  "Colour: Activation  (red: low, green: high)    Opacity: Weight/Bias",
                  ha="center"
                  )
        self.axis("off")
        self.set_aspect("equal")
        self.set_title("Multilayer Perceptron", size=20)


class MultilayerPerceptron:
    """
    The class for the multilayer perceptron.

    Attributes
    ----------
    sizes: list of int
        A list of the number of perceptrons in each layer.
    activations: list of ndarray(dtype=np.float32, ndim=1)
        A list of arrays storing the activation of each perceptron.
        Accessed with activations[layer][index].
    biases: list of ndarray(dtype=np.float32, ndim=1)
        A list of arrays storing the bias of each perceptron.
        Accessed with biases[layer][index].
    weights: list of ndarray(dtype=np.float32, ndim=2)
        A list of arrays storing the weights of each connection.
        The length is one less than activations because the input and output layers only have one connection.
        Accessed with weights[layer of left neuron][index of left neuron][index of right neuron]

    Methods
    -------
    display(max_height=None, layer_width=6, output_labels=None)
        Display a diagram of the multilayer perceptron using a `Diagram` object.
    calculate_activations()
        Calculate the activations of neurons in layers outside the input layer.
    """

    # Initialise the class with activations, biases and weights based on the entered size array
    def __init__(self, sizes):
        """
        Initialise the `MultilayerPerceptron` object with a given size.

        Parameters
        ----------
        sizes: list of integers
            A list of the number of perceptrons in each layer.
        """
        self.sizes = sizes
        # Instance a generator object to create random numbers
        random_gen = np.random.default_rng(42)
        # Create lists of arrays of random numbers
        self.activations = [random_gen.random(i, dtype=np.float32) for i in sizes]
        self.biases = [10 * random_gen.random(i, dtype=np.float32) - 5 for i in sizes[1:]]
        self.weights = [10 * random_gen.random((sizes[i], sizes[i + 1])) - 5 for i in range(len(sizes) - 1)]

    def display(self, max_height=None, layer_width=6, output_labels=None):
        """
        Displays a diagram of the `MultilayerPerceptron` with a given look.

        Parameters
        ----------
        max_height: int, default=None
            The maximum number of perceptrons shown in one layer.
        layer_width: int, default=6
            The width of each layer.
        output_labels: list of str, default=None
            A list containing the names of the output perceptrons.
        """
        # Ensure the parameters are valid
        assert isinstance(max_height, (int, float)) and max_height > 0, "max_height must be a positive, real number."
        assert isinstance(layer_width, (int, float)) and layer_width > 0, "layer_width must be a positive, real number."
        assert output_labels is None or (isinstance(output_labels, list) and all(
            isinstance(label, str) for label in output_labels)), "output_labels must be None or list of strings."

        # Create a figure for the diagram
        fig = plt.figure(figsize=(12, 8))
        # Create a Diagram object using the parameters and the figure
        diagram = Diagram(self, fig, max_height, layer_width, output_labels)

        # Show the diagram
        diagram.show_diagram()
        plt.show()

    def calculate_activations(self):
        """Calculate the activations of neurons in layers outside the input layer."""
        # Loop for each layer
        for x in range(1, len(self.sizes)):
            # Calculate the new activations of that layer in parallel
            self.activations[x] = sigmoid(self.activations[x - 1] @ self.weights[x - 1] + self.biases[x - 1])


MLP = MultilayerPerceptron([748, 16, 16, 16, 10])
MLP.calculate_activations()
MLP.display(max_height=16, output_labels=[str(i) for i in range(1, 11)])
# MLP.display(max_height=16, output_labels=[str(i) for i in range(1, 11)])

(training_images, training_outputs), (testing_images, testing_outputs) = load_dataset("./mnist-dataset")
