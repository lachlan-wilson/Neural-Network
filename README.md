# Neural Network
## Overview
This is a machine learning algorithm that performs optical character recognition on the digits 0-9 and has been trained on the MNIST database.

When I set out to complete this project my goal was to understand how machine learning algorithms work. A multilayer perceptron or deep neural network is a perfect algorithm for this goal as it is complex enough to not be intuitive but not too complex to be incomprehensible.
I think that you learn best by doing, and so I didn't follow a tutorial or look at code for other DNNs but instead followed 3Blue1Brown's series on DNNs which explores the theory behind them. This forced me to develop a real understanding of the algorithm because I had to code it from scratch.
Developing this project really outlined how "simple" ML algorithms are, it's clear to me now that even the most complex algorithms are still just maths.

## Example

## Usage
As the goal of this project was for my understanding and not the use of others it has not been designed with that in mind, that said it is not complex to use but rather is missing some QofL features.
It should be intuitive and I have implemented input validation on most things. 

Simply run:
   ```bash
   main.py
   ```

**The only thing to be aware of is that loaded .npz files must match that of the initialisation**
I use a shape of [784, 16, 16, 10] for my MLPs but the only important values for the MNIST dataset are the input and output layers. These must be 784 and 10 as there are 784 pixels in the image and 10 possible digits.

## Installation
### Dependencies
- NumPy
- matplotlib
- 
### Environment Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/lachlan-wilson/Neural-Network.git
    cd Neural-Network
    ```



