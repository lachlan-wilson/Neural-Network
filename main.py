# ----- Multilayer Perceptron -----
#       ----- 31/03/2026 -----

# External libraries
import matplotlib
matplotlib.use("TkAgg")
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
    Follows function σ(x) = 1/(1+e^-x)

    Parameters
    ----------
    x: ndarray(dtype=float, ndim=1) or float
        The value(s) that will be put through the function.

    Returns
    -------
    ndarray(dtype=float, ndim=1) or float
        The output(s) of the function
    """
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    The derivative of the sigmoid function.
    Follow function σ'(x) = σ(x) * (1-σ(x))

    Parameters
    ----------
    x: ndarray(dtype=float, ndim=1) or float
        The value(s) that will be put through the function.

    Returns
    -------
    ndarray(dtype=float, ndim=1) or float
        The output(s) of the function
    """
    s = sigmoid(x)
    return s * (1 - s)


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
        A dictionary containing data about neurons so that they can be updated.
    paths: dict
        A dictionary containing data about paths so that they can be updated.
    cost: `Text` object
        An object for the cost of an epoch.

    Methods
    -------
    show_diagram()
        Display the diagram in its `Figure`.
    update_diagram()
        Update the diagram based on new dataset index and new activations.
    """

    def __init__(self, mlp, __figure, max_height=None, layer_width=6, output_labels=None, **kwargs):
        """
        Initialise the `Diagram` object with the given parameters. Assigns itself to the global __figure.

        Parameters
        ----------
        mlp: NewMultilayerPerceptron
            An instance of the `MultilayerPerceptron` object.
        __figure: Figure
            The `Figure` object the diagram should be contained in.
        max_height: int, default=None
            The maximum number of neurons shown in one layer.
        layer_width: int, default=6
            The width of each layer.
        output_labels: list of str, default=None
            A list containing the names of the output neurons.
        **kwargs: dict
            Extra arguments to `Figure`: refer to `Figure` documentation for a
            list of all possible arguments.
        """
        # Ensure all the parameters are valid
        assert isinstance(mlp, NewMultilayerPerceptron), "mlp must be a MultilayerPerceptron."
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
        self.cost = None

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
                          linewidth=colour[-1] * 1.5,
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
                  f"{self.mlp.sizes[x]}",
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
        """Display the plot."""
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
        # self.cost = self._show_cost()
        self.axis("off")
        self.grid("off")
        self.set_aspect("equal")
        self.set_title("Multilayer Perceptron", size=60)

    def update_diagram(self):
        """Update the diagram based on new dataset index and new activations."""
        for (layer, neuron_index), neuron in self.neurons.items():
            colour = (1 - self.mlp.activations[layer][neuron_index],  # Red
                      self.mlp.activations[layer][neuron_index],  # Green
                      0,  # Blue
                      1 if layer == 0 else non_linear(  # Alpha (Opacity)
                          sigmoid(self.mlp.biases[layer - 1][neuron_index]))
                      )
            neuron.set_facecolor(colour)
            neuron.set_edgecolor((*colour[:-1], 1))
            neuron.stale = True

        for (layer, neuron_index, next_neuron_index), path in self.paths.items():
            colour = (1 - self.mlp.activations[layer][neuron_index],
                      self.mlp.activations[layer][neuron_index],
                      0,
                      non_linear(sigmoid(self.mlp.weights[layer][neuron_index][next_neuron_index]))
                      )
            path.set_color(colour)
            path.set_linewidth(colour[-1] * 1.5)
            path.stale = True

        # self.cost.set_text(f"Cost: {self.mlp.cost()}")


# class MultilayerPerceptron:
#     """
#     The class for the multilayer neuron.
#
#     Attributes
#     ----------
#     sizes: list of int
#         A list of the number of perceptrons in each layer.
#     activations: list of ndarray(dtype=np.float32, ndim=1)
#         A list of arrays storing the activation of each neuron.
#         Accessed with activations[layer][index].
#     weighted_inputs: list of ndarray(dtype=np.float32, ndim=1)
#         A list of arrays storing the weighted input to each neuron (activation before sigmoid function).
#         Accessed with activations[layer][index].
#     biases: list of ndarray(dtype=np.float32, ndim=1)
#         A list of arrays storing the bias of each neuron.
#         Accessed with biases[layer][index].
#     weights: list of ndarray(dtype=np.float32, ndim=2)
#         A list of arrays storing the weights of each connection.
#         The length is one less than activations because the input and output layers only have one connection.
#         Accessed with weights[layer of left neuron][index of left neuron][index of right neuron].
#     __figure: None or `Figure` object
#         The `Figure` that will contain the diagram and the image.
#     __diagram: None or `Diagram(Axes)` object
#         The `Diagram` of the mlp.
#     __image_axes: None or `Axes` object
#         The `Axes` that will contain the image
#     __image_object: `Image`
#         The `Image` object that contains the image object for updating
#     image: array-like image
#         The image of the number for a given dataset index
#     label: str
#         The label for the image of the number for a given dataset index
#     costs: None or ndarray(ndim = 1)
#         An array of costs of for each dataset index
#
#     Methods
#     -------
#     display(max_height=None, layer_width=6, output_labels=None)
#         Display a diagram of the multilayer perceptron using a `Diagram` object.
#     update_display(self, dataset, data_index=0)
#         Update the display based on new dataset index and new activations.
#     calculate_activations()
#         Calculate the activations of neurons in layers outside the input layer.
#     use_input(self, dataset, index=0)
#         Update the input layer to use the data.
#     cost()
#         Calculate the cost of this dataset index.
#     average_cost(self, dataset)
#         Calculates the average cost over the full dataset
#     """
#
#     # Initialise the class with activations, biases and weights based on the entered size array
#     def __init__(self, sizes, dataset):
#         """
#         Initialise the `MultilayerPerceptron` object with a given size.
#
#         Parameters
#         ----------
#         sizes: list of integers
#             A list of the number of perceptrons in each layer.
#         dataset: tuple[Any, Union[array, array[Union[int, float, str]]]]
#             The data of the MNIST dataset.
#         """
#         self.sizes = sizes
#         self.dataset = dataset
#         # Instance a generator object to create random numbers
#         random_gen = np.random.default_rng(42)
#         # Create lists of arrays of random numbers
#         self.activations = [random_gen.random(size, dtype=np.float32) for size in sizes]
#         self.weighted_inputs = [random_gen.random(size, dtype=np.float32) for size in sizes[1:]]
#         # self.activations = [np.linspace(0.05, 0.95, size, dtype=np.float32) for size in sizes]
#         self.biases = [10 * random_gen.random(size, dtype=np.float32) - 5 for size in sizes[1:]]
#         self.weights = [10 * random_gen.random((sizes[i], sizes[i + 1])) - 5 for i in range(len(sizes) - 1)]
#
#         self.__figure = None
#         self.__image_axes = None
#         self.__diagram = None
#         self.__image_object = None
#
#         self.image = None
#         self.label = None
#
#         self.costs = None
#
#     def show_image(self):
#         """Display an MNIST image using matplotlib."""
#         # Create a new image if there isn't one
#         if self.__image_object is None:
#             self.__image_object = self.__image_axes.imshow(self.image, cmap="gray")
#         else:
#             self.__image_object.set_data(self.image)
#
#         self.__image_axes.set_title(f"Label: {self.label}")
#         self.__image_axes.axis("off")
#
#     def display(self, max_height=None, layer_width=6, output_labels=None):
#         """
#         Displays a diagram of the `MultilayerPerceptron` with a given look.
#
#         Parameters
#         ----------
#         max_height: int, default=None
#             The maximum number of perceptrons shown in one layer.
#         layer_width: int, default=6
#             The width of each layer.
#         output_labels: list of str, default=None
#             A list containing the names of the output perceptrons.
#         """
#         # Ensure the parameters are valid
#         assert max_height is None or isinstance(max_height, (
#             int, float)) and max_height > 0, "max_height must be None or a positive, real number."
#         assert isinstance(layer_width, (int, float)) and layer_width > 0, "layer_width must be a positive, real number."
#         assert output_labels is None or (isinstance(output_labels, list) and all(
#             isinstance(label, str) for label in output_labels)), "output_labels must be None or list of strings."
#
#         # If max height wasn't defined set it to the biggest layer size
#         if max_height is None or max_height > max(self.sizes):
#             max_height = max(self.sizes)
#         else:
#             max_height = max_height
#
#         self.use_input()
#
#         # Create a figure for the diagram
#         self.__figure = plt.figure(figsize=(len(self.sizes) * layer_width + 10, max_height + 10))
#         # Create a Diagram object using the parameters and the __figure
#         self.__diagram = Diagram(self, self.__figure, max_height, layer_width, output_labels)
#
#         # Show the diagram
#         self.__diagram.show_diagram()
#
#         self.__image_axes = plt.Axes(self.__figure, [0.05, 0.35, 0.2, 0.3])
#         self.__figure.add_axes(self.__image_axes)
#         self.show_image()
#
#     def update_display(self, dataset=None, data_index=0):
#         """
#         Update the display based on new dataset index and new activations.
#
#         Parameters
#         ----------
#         dataset: None or tuple[Any, Union[array, array[Union[int, float, str]]]], default=None
#             A section of the MNIST dataset
#         data_index: int, default=0
#             The index of the data to be displayed.
#         """
#         if dataset is None:
#             dataset = self.dataset
#         # Update the MLP
#         self.use_input(dataset, data_index)
#
#         # Update the diagram
#         self.__diagram.update_diagram()
#
#         # Update the image
#         self.show_image()
#
#         # Refresh the canvas
#         self.__figure.stale = True
#         self.__figure.canvas.draw()
#         self.__figure.canvas.flush_events()
#         plt.pause(0.001)
#         plt.draw()
#
#     def calculate_activations(self):
#         """Calculate the activations of neurons in layers outside the input layer."""
#         # Loop for each layer
#         for x in range(1, len(self.sizes)):
#             # Calculate the new activations of that layer in parallel
#             self.weighted_inputs[x - 1] = self.activations[x - 1] @ self.weights[x - 1] + self.biases[x - 1]
#             self.activations[x] = sigmoid(self.weighted_inputs[x - 1])
#
#     def use_input(self, image=None, label=None):
#         """
#         Update the input layer to use the data.
#
#         Parameters
#         ----------
#         image: None or array-like-image:
#             The image of the number to be used.
#         label: None or int
#             The label of the number to be used.
#         """
#         if image is None:
#             image = self.dataset[0][0]
#
#         if label is None:
#             label = self.dataset[1][0]
#
#         self.image = image
#         self.label = label
#         self.activations[0] = 1 - image.flatten().astype(np.float32)
#         self.calculate_activations()
#
#     def cost(self):
#         """
#         Calculate the cost of this dataset index.
#
#         Returns
#         -------
#         cost: int
#             cost of this dataset index.
#         """
#         answer = np.zeros(self.sizes[-1], dtype=np.float32)
#         answer[self.label] = 1
#         return ((self.activations[-1] - answer) ** 2).sum()
#
#     def average_cost(self, batch):
#         self.costs = np.zeros(len(batch[0]))
#
#         for i in range(len(batch[0])):
#             self.use_input(batch, i)
#             self.costs[i] = self.cost()
#         return np.mean(self.costs)
#
#     def run_stochastic_gradient_descent(self):
#         images = np.array_split(self.dataset[0], int(len(self.dataset[0]) / 100))
#         labels = np.array_split(self.dataset[1], int(len(self.dataset[1]) / 100))
#
#         for image_batch, label_batch in zip(images, labels):
#             for image, label in zip(image_batch, label_batch):
#                 self.use_input(image, label)
#                 self.calculate_activations()
#                 cost = self.cost()


class NewMultilayerPerceptron:

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
        self.activations = [np.zeros(size, dtype=np.float32) for size in sizes]
        self.biases = [np.zeros(size, dtype=np.float32) for size in sizes[1:]]
        self.weights = [(random_gen.random((sizes[i], sizes[i + 1]), dtype=np.float32) - 0.5) * 20 for i in range(len(sizes) - 1)]

        # Initialises variables to store objects needed for the diagram
        self.__figure = None
        self.__image_axes = None
        self.__diagram = None
        self.__image_object = None

    def _forward_pass(self, image):
        """
        Calculate the activations of each neuron based on the input neurons from the image.

        Parameters
        ----------
        image: array-like image
            The 28x28 image of a number.

        Returns
        -------
        weighted_inputs: list of ndarray(dtype=np.float32, ndim=1)
            The inputs each neuron before they're based through the sigmoid function.
            Uses when training the model, often called `z`.
        """
        # Set the input layer to match the image
        self.activations[0] = 1 - image.flatten().astype(np.float32)

        # The weighted input (activation before sigmoid) for each neuron not in the input layer
        weighted_inputs = [np.zeros(size, dtype=np.float32) for size in self.sizes[1:]]

        # Loop for each layer except the input layer
        for x in range(1, len(self.sizes)):
            # z^L = A^L-1 * W^L + b^L is repeated for all neurons
            # Where L is the layer,
            # A is a vector of all the activations in the previous layer and
            # W is a vector of the weights for this neuron
            weighted_inputs[x - 1] = self.activations[x - 1] @ self.weights[x - 1] + self.biases[x - 1]
            # a^L = σ(z^L)
            self.activations[x] = sigmoid(weighted_inputs[x - 1])

        return weighted_inputs

    def train(self, dataset, learning_rate, batch_size=100):
        """
        Train the network, looping through all the training data once (1 epoch).
        Updates the weights and biases of the network to better classify digits.

        Parameters
        ----------
        dataset: tuple[Any, Union[array, array[Union[int, float, str]]]]
            A tuple of arrays of images and labels.
        learning_rate: float
            The size of a gradient descent step.
        batch_size: int
            The size of a mini-batch
        """

        def backpropagation(weighted_inputs, label):
            """
            Runs the backpropogation algorithm on the current networks activations and the desired output.

            Parameters
            ----------
            weighted_inputs: list of ndarray(dtype=np.float32, ndim=1)
                The inputs each neuron before they're based through the sigmoid function.
                Uses when training the model, often called `z`.
            label: int
                The desired "guess" of the network.

            Returns
            -------
            weight_gradients: list of ndarray(dtype=np.float32, ndim=1)
                A vector of the changes that should be made to weights for this pass.
            bias_gradients: list of ndarray(dtype=np.float32, ndim=2)
                A vector of the changes that should be made to biases for this pass.
            """

            # Vector of the desired output (y)
            desired_output = np.zeros(self.sizes[-1], dtype=np.float32)
            desired_output[label] = 1.0

            # Initialise vectors to store the changes in weights and biases
            weight_gradients = [np.zeros_like(w) for w in self.weights]
            bias_gradients = [np.zeros_like(b) for b in self.biases]

            # # Cost: C = sum((a-y)^2
            # # Partial derivative dC/da: 2(a-y)
            # # Where a is the output layer activations vector
            # # and y is the output vector
            # dC_da_output = 2 * (self.activations[-1] - desired_output)
            #
            # # Partial derivative da/dz: σ'(z)
            # da_dz_output = sigmoid_derivative(weighted_inputs[-1])
            #
            # # Intermediate step to getting the changes for bias and weights
            # layer_delta = dC_da_output * da_dz_output

            layer_delta = self.activations[-1] - desired_output

            # Storing the wanted changes in biases and weights
            bias_gradients[-1] = layer_delta
            weight_gradients[-1] = np.outer(self.activations[-2], layer_delta)

            # Looping backwards through the layers
            for layer_offset in range(2, len(self.sizes)):
                # Similar to above but for the previous layers
                da_dz = sigmoid_derivative(weighted_inputs[-layer_offset])

                layer_delta = (self.weights[-layer_offset + 1] @ layer_delta) * da_dz

                bias_gradients[-layer_offset] = layer_delta
                weight_gradients[-layer_offset] = np.outer(self.activations[-layer_offset - 1], layer_delta)

            return weight_gradients, bias_gradients

        images = dataset[0]
        labels = dataset[1]
        num_samples = len(images)

        # Shuffle images and labels
        permutation = np.random.permutation(num_samples).tolist()
        shuffled_images = images[permutation]
        shuffled_labels = [labels[i] for i in permutation]

        # Split images and labels into mini-batches for efficiency
        image_batches = np.array_split(shuffled_images, num_samples / batch_size)
        label_batches = np.array_split(shuffled_labels, num_samples / batch_size)

        # Loop for each image-label pair
        for image_batch, label_batch in zip(image_batches, label_batches):
            # Initialise vectors for the total changes and weights for this batch
            summed_weight_gradients = [np.zeros_like(w, dtype=np.float32) for w in self.weights]
            summed_bias_gradients = [np.zeros_like(b, dtype=np.float32) for b in self.biases]

            # For each image and label in the batch
            for image, label in zip(image_batch, label_batch):
                # Update the activations for that image
                weighted_inputs = self._forward_pass(image)
                # Calculate the vectors for the changes in weights and biases
                weight_gradients, bias_gradients = backpropagation(weighted_inputs, label)

                # Add these to the total changes
                for layer in range(len(self.sizes) - 1):
                    summed_weight_gradients[layer] += weight_gradients[layer]
                    summed_bias_gradients[layer] += bias_gradients[layer]

            # Average the changes and then apply them
            for layer in range(len(self.sizes) - 1):
                self.weights[layer] -= learning_rate * (summed_weight_gradients[layer] / len(image_batch))
                self.biases[layer] -= learning_rate * (summed_bias_gradients[layer] / len(image_batch))

        total_cost = 0
        for image, label in zip(images, labels):
            _ = self._forward_pass(image)
            answer = np.zeros(self.sizes[-1], dtype=np.float32)
            answer[label] = 1
            total_cost += ((self.activations[-1] - answer) ** 2).sum()
        print(f"Cost: {total_cost / num_samples}")

    def test(self, dataset):
        """
        Tests the network based on the data.

        Parameters
        ----------
        dataset: dataset: tuple[Any, Union[array, array[Union[int, float, str]]]]
            A tuple of arrays of images and labels.

        Returns
        -------
        float
            The accuracy of the network as a decimal.
        """
        # Initialize the variables
        images = dataset[0]
        labels = dataset[1]
        correct = 0
        answers = [0 for i in range(10)]
        total = len(images)

        # Loop for each image-label pair
        for image, label in zip(images, labels):
            # Update the network with the image
            self._forward_pass(image)
            # print(self.activations[-1])

            # If the highest output from the network is correct
            if label == np.argmax(self.activations[-1]):
                # Increment correct
                correct += 1

            # answers[np.argmax(self.activations[-1])] += 1

        return correct / total

    def create_display(self, max_height=None, layer_width=6, output_labels=None):
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

        self.__image_axes = plt.Axes(self.__figure, [0.05, 0.35, 0.2, 0.3])
        self.__figure.add_axes(self.__image_axes)

        plt.show()

    def display(self, dataset, index=0):
        _ = self._forward_pass(dataset[0][index])

        self.__diagram.update_diagram()

        self._show_image(dataset[0][index], dataset[1][index])
        self.__diagram.show_diagram()

        self.__figure.stale = True
        self.__figure.canvas.draw()
        self.__figure.canvas.flush_events()
        plt.pause(0.001)
        plt.draw()

    def _show_image(self, image, label):
        if self.__image_object is None:
            self.__image_object = self.__image_axes.imshow(image, cmap="gray")
        else:
            self.__image_object.set_data(image)
        self.__image_axes.set_title(f"Label: {label}")
        self.__image_axes.axis("off")


train_mnist = mnist_reader.MNIST()
data = train_mnist.load()

test_mnist = mnist_reader.MNIST(name_img="t10k-images.idx3-ubyte", name_lbl="t10k-labels.idx1-ubyte")
test_data = test_mnist.load()

MLP = NewMultilayerPerceptron([784, 16, 16, 10])

MLP.create_display(16, 6, [str(i) for i in range(10)])

MLP.display(data)

# print(f"Accuracy: {round(MLP.test(test_data), 4) * 100}%")
# for i in range(400):
#     print(f"Epoch: {i}/400")
#     MLP.train(data, 0.05)
# print(f"Accuracy: {round(MLP.test(test_data), 4) * 100}%")
