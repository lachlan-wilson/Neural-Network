# ----- Multilayer Perceptron -----
#       ----- 31/03/2026 -----

# External libraries
import matplotlib.pyplot as plt
import numpy as np

# Internal libraries
import mnist_reader


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
        The maximum number of neurons shown in one layer.
    layer_width: int
        The width of each layer.
    x_offset: float
        The x shift of each neuron, `layer_width / 2`.
    output_labels: list of str
        A list containing the names of the output neurons.
    neuron_coordinates: list of ndarray(dtype=float, nDim=2)
        A list containing arrays of the coordinates of each neuron.
    neurons: dict
        A dictionary containing data about neurons so that they can be updated
    paths: dict
        A dictionary containing data about paths so that they can be updated
    Methods
    -------
    show_diagram()
        Display the diagram in its `Figure`
    """

    def __init__(self, mlp, __figure, max_height=None, layer_width=6, output_labels=None, **kwargs):
        """
        Initialise the `Diagram` object with the given parameters. Assigns itself to the global __figure.

        Parameters
        ----------
        mlp: MultilayerPerceptron
            An instance of the `MultilayerPerceptron` object.
        __figure: Figure
            The `Figure` object the diagram should be contained in.
        max_height: int, default=None
            The maximum number of neurons shown in one layer.
        layer_width: int, default=6
            The width of each layer.
        output_labels: list of str, default=
            A list containing the names of the output neurons
        **kwargs: dict
            Extra arguments to `Figure`: refer to `Figure` documentation for a
            list of all possible arguments.
        """
        # Ensure all the parameters are valid
        assert isinstance(mlp, MultilayerPerceptron), "mlp must be a MultilayerPerceptron."
        assert isinstance(__figure, plt.Figure), "__figure must be a Figure."

        super().__init__(__figure, [0.3, 0.1, 0.65, 0.8], **kwargs)
        __figure.add_axes(self)

        # Store variables as class attributes
        self.__figure = __figure
        self.mlp = mlp
        self.layer_width = layer_width
        self.x_offset = layer_width / 2
        self.output_labels = output_labels
        self.max_height = max_height

        # Store coordinates of neurons
        self.neuron_coordinates = [np.zeros((size, 2)) for size in self.mlp.sizes]
        # Loop for each layer with an index of x
        for x, layer in enumerate(self.neuron_coordinates):
            layer_height = self.mlp.sizes[x]

            # Set the x-ordinate of the neurons in this layer
            layer[:, 0] = layer_width * (x + 0.5)

            y_offset = (max_height - layer_height) / 2

            # If the layer needs truncated
            if layer_height > max_height:
                y_offset = 0
                # Set the y-ordinate of neurons to be hidden to -1
                layer[max_height // 2 - 1:layer_height - max_height // 2 + 1, 1] = -1

                # Create a mask of where the y-ordinate is not -1
                mask = layer[:, 1] != -1
                # Create a range for each neuron in the mask
                indices = np.arange(np.sum(mask))
                # Add 2 to the indices above the halfway point
                indices[np.sum(mask) // 2:] += 2 + (max_height & 1)
                # Apply the indices to the y values in the layer
                layer[mask, 1] = indices + y_offset + 1
            else:
                # Set the y value of the neurons to its index and centre it
                indices = np.arange(layer_height)
                layer[:, 1] = indices + y_offset + 1

        # Store the objects for updating
        self.neurons = {}
        self.paths = {}

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

    @staticmethod
    def _circle_object(centre, colour=(1, 1, 1, 1)):
        """
        Create an object of a circle at a given position.

        Parameters
        ----------
        centre: tuple of float
            Coordinate (x, y) of the centre of the circle.
        colour: tuple of float, default=(1, 1, 1, 1) (white)
            RBGA value of the colour of the circle.

        Returns
        -------
        `Circle`
            A circle object with given properties, can be updated later.
        """
        return plt.Circle(centre,
                          0.4,
                          edgecolor=(colour[:-1], 1),
                          facecolor=colour,
                          zorder=10  # Add circles above lines
                          )

    def _add_line_(self, xs, ys, colour):
        """
        Add a line to the plot at given positions and return the line object for updating.

        Parameters
        ----------
        xs: list of float
            List of the x-ordinates [x1, x2].
        ys: list of float
            List of the y-ordinates [y1, y2].
        colour: tuple of float
            RGBA value of the colour of the line.

        Returns
        -------
        `Line2D`
            A line object with given properties, can be updated later.
        """
        # "," unpacks the tuple of a single Line object into only the line
        line, = self.plot(xs,
                          ys,
                          color=colour,
                          linewidth=colour[-1] * 3,
                          zorder=1
                          )
        return line

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

    def _add_neuron(self, layer, neuron_index):
        """
        Add a neuron to the plot at a given position.

        Parameters
        ----------
        layer: int
            The layer number for this neurons layer
        neuron_index: int
            The neuron's position within the layer
        """
        if self.neuron_coordinates[layer][neuron_index][1] != -1:
            neuron_centre = tuple(self.neuron_coordinates[layer][neuron_index, :])
            # Add a solid background circle
            self.add_patch(self._circle_object(neuron_centre))
            # Add a circle in the same position where the colour represents the activation of the neuron and opacity the bias
            neuron = self._circle_object(neuron_centre,
                                         (1 - self.mlp.activations[layer][neuron_index],  # Red
                                          self.mlp.activations[layer][neuron_index],  # Green
                                          0,  # Blue
                                          1 if layer == 0 else non_linear(  # Alpha (Opacity)
                                              sigmoid(self.mlp.biases[layer - 1][neuron_index]))
                                          ))
            self.add_patch(neuron)
            self.neurons[(layer, neuron_index)] = neuron

            # If it's not the output layer
            if layer < len(self.mlp.sizes) - 1:
                # Loop for each neuron in the next layer
                for next_neuron_index in range(self.mlp.sizes[layer + 1]):
                    if self.neuron_coordinates[layer + 1][next_neuron_index][1] != -1:
                        # Draw a line between the centre of the two perceptrons
                        line = self._add_line_(
                            [neuron_centre[0], self.neuron_coordinates[layer + 1][next_neuron_index][0]],
                            [neuron_centre[1], self.neuron_coordinates[layer + 1][next_neuron_index][1]],
                            (1 - self.mlp.activations[layer][neuron_index],
                             self.mlp.activations[layer][neuron_index],
                             0,
                             non_linear(sigmoid(self.mlp.weights[layer][neuron_index][next_neuron_index]))
                             ))
                        self.paths[(layer, neuron_index, next_neuron_index)] = line

    def show_diagram(self):
        """
        Display the plot.
        """
        # Add the background boxes
        self._add_box((0, 0), self.layer_width, "Input Layer")
        self._add_box((self.layer_width, 0), self.layer_width * (len(self.mlp.sizes) - 2), "Hidden Layer")
        self._add_box((self.layer_width * (len(self.mlp.sizes) - 1), 0), self.layer_width, "Output Layer")

        # Loop for each layer
        for layer in range(len(self.mlp.sizes)):

            self._add_layer_title(layer)

            for neuron_index in range(self.mlp.sizes[layer]):
                self._add_neuron(layer, neuron_index)

            if self.mlp.sizes[layer] > self.max_height:
                for i in [-0.4, 0, 0.4]:
                    self.add_patch(plt.Circle((layer * self.layer_width + self.x_offset, (self.max_height + 1) / 2 + i),
                                              0.08,
                                              color="black"
                                              ))

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
                  ha="center",
                  clip_on=False
                  )
        self.axis("off")
        self.grid("off")
        self.set_aspect("equal")
        self.set_title("Multilayer Perceptron", size=60)

    def update_diagram(self):
        """
        Update the diagram to show the new activations.
        """
        for (layer, neuron_index), neuron in self.neurons.items():
            colour = (1 - self.mlp.activations[layer][neuron_index],  # Red
                      self.mlp.activations[layer][neuron_index],  # Green
                      0,  # Blue
                      1 if layer == 0 else non_linear(  # Alpha (Opacity)
                          sigmoid(self.mlp.biases[layer - 1][neuron_index]))
                      )
            neuron.set_facecolor(colour)
            neuron.set_edgecolor((*colour[:-1], 1))

        for (layer, neuron_index, next_neuron_index), path in self.paths.items():
            colour = (1 - self.mlp.activations[layer][neuron_index],
                      self.mlp.activations[layer][neuron_index],
                      0,
                      non_linear(sigmoid(self.mlp.weights[layer][neuron_index][next_neuron_index]))
                      )
            path.set_color(colour)
            path.set_linewidth(colour[-1] * 3)


class MultilayerPerceptron:
    """
    The class for the multilayer neuron.

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
        Accessed with weights[layer of left neuron][index of left neuron][index of right neuron].
    __figure: None or `Figure` object
        The `Figure` that will contain the diagram and the image.
    __diagram: None or `Diagram(Axes)` object
        The `Diagram` of the mlp.
    __image_axes: None or `Axes` object
        The `Axes` that will contain the image
    __image_object: None or array-like image
        The `array-like image` that contains the image object for updating

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
        self.activations = [random_gen.random(size, dtype=np.float32) for size in sizes]
        # self.activations = [np.linspace(0.05, 0.95, size, dtype=np.float32) for size in sizes]
        self.biases = [10 * random_gen.random(size, dtype=np.float32) - 5 for size in sizes[1:]]
        self.weights = [10 * random_gen.random((sizes[i], sizes[i + 1])) - 5 for i in range(len(sizes) - 1)]

        self.__figure = None
        self.__image_axes = None
        self.__diagram = None
        self.__image_object = None

    def show_image(self, dataset, data_index=0):
        """
        Display an MNIST image using matplotlib.

        Parameters
        ----------
        dataset: tuple[Any, Union[array, array[Union[int, float, str]]]]
            The MNIST dataset data.
        data_index: int, default=0
            The index of the data to be shown.
        """
        # Create a new image if there isn't one
        if self.__image_object is None:
            self.__image_object = self.__image_axes.imshow(dataset[0][data_index], cmap="gray")
        else:
            self.__image_object.set_data(dataset[0][data_index])

        self.__image_axes.set_title(f"Label: {dataset[1][data_index]}")
        self.__image_axes.axis("off")

    def display(self, dataset, max_height=None, layer_width=6, output_labels=None):
        """
        Displays a diagram of the `MultilayerPerceptron` with a given look.

        Parameters
        ----------
        dataset: tuple of list
            The dataset to be used.
        max_height: int, default=None
            The maximum number of perceptrons shown in one layer.
        layer_width: int, default=6
            The width of each layer.
        output_labels: list of str, default=None
            A list containing the names of the output perceptrons.
        """
        # Ensure the parameters are valid
        assert max_height is None or isinstance(max_height, (
            int, float)) and max_height > 0, "max_height must be None or a positive, real number."
        assert isinstance(layer_width, (int, float)) and layer_width > 0, "layer_width must be a positive, real number."
        assert output_labels is None or (isinstance(output_labels, list) and all(
            isinstance(label, str) for label in output_labels)), "output_labels must be None or list of strings."

        # If max height wasn't defined set it to the biggest layer size
        if max_height is None or max_height > max(self.sizes):
            max_height = max(self.sizes)
        else:
            max_height = max_height

        # Create a figure for the diagram
        self.__figure = plt.figure(figsize=(len(self.sizes) * layer_width + 10, max_height + 10))
        # Create a Diagram object using the parameters and the __figure
        self.__diagram = Diagram(self, self.__figure, max_height, layer_width, output_labels)

        # Show the diagram
        self.__diagram.show_diagram()

        self.__image_axes = plt.Axes(self.__figure, [0.05, 0.35, 0.2, 0.3])
        self.__figure.add_axes(self.__image_axes)
        self.show_image(dataset, 0)

    def update_display(self, dataset, data_index=0):
        # Update the MLP
        self.use_input(dataset[0][data_index])

        # Update the diagram
        self.__diagram.update_diagram()

        # Update the image
        self.show_image(dataset, data_index)

        # Refresh the canvas
        self.__figure.canvas.draw()
        self.__figure.canvas.flush_events()
        # Pause allowing the window to refresh
        plt.pause(0.1)

    def calculate_activations(self):
        """Calculate the activations of neurons in layers outside the input layer."""
        # Loop for each layer
        for x in range(1, len(self.sizes)):
            # Calculate the new activations of that layer in parallel
            self.activations[x] = sigmoid(self.activations[x - 1] @ self.weights[x - 1] + self.biases[x - 1])

    def use_input(self, image):
        self.activations[0] = 1 - image.flatten().astype(np.float32)
        self.calculate_activations()


plt.style.use("./style.mlpstyle")  # Use the styles located at ./styles.mlpstyle
plt.ion()

MLP = MultilayerPerceptron([784, 16, 16, 16, 10])

train_mnist = mnist_reader.MNIST()
data = train_mnist.load()
MLP.calculate_activations()
MLP.display(data, max_height=16, output_labels=[str(i) for i in range(1, 11)])

MLP.update_display(data, 10)
MLP.update_display(data, 11)
plt.ioff()
plt.show()
