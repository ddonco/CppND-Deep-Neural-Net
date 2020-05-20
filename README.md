# CPPND: Capstone - Deep Neural Netowork

This capstone project implements a simple deep neural network for training and inferencing on tabular data. A user defined network of fully connected layers can be trained using stochastic gradient descent as the optimizer and categorical cross-entropy as the loss function. This project serves as an exercise to both solidify a multitude of concepts regarding programming in C++ as well as gain a deeper understanding of the mechanics within deep neural networks.

## Dependencies for Running Locally

- cmake >= 3.7
  - All OSes: [click here for installation instructions](https://cmake.org/install/)
- make >= 4.1 (Linux, Mac), 3.81 (Windows)
  - Linux: make is installed by default on most Linux distros
  - Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  - Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
- gcc/g++ >= 5.4
  - Linux: gcc / g++ is installed by default on most Linux distros
  - Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  - Windows: recommend using [MinGW](http://www.mingw.org/)
- eigen >= 3.3.7
  1. `git clone https://gitlab.com/libeigen/eigen.git` to workspace
  2. `cd eigen`
  3. `mkdir build && cd build`
  4. `cmake ..`
  5. `make install`
- matplotlib-cpp
  - C++ wrapper is already included in project source but can also be cloned from github [https://github.com/lava/matplotlib-cpp]
  - requirements for matplotlib-cpp can be satisfied by running: `sudo apt-get install python-matplotlib python-numpy python2.7-dev`

## Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./DNN train ../config/l3.config ../data/test.weights ../data/X.csv ../data/Y.csv`

## Usage

### Training

Like any machine learning model, a deep neural network must first be trained before it can be useful. A config file must be defined to build the neural network. A sample config file can be found in the `config` folder. The number of training epochs and learning rate are set at the top of the file under `[train]`. The model configuration is specified below the training parameters.

At this time the supported layer and activation types are:

- Layers
  - dense
- Activations
  - relu
  - softmax

A model weights path must be specified when training to provide a destination for the final model weights after training. If you would like to starting trainig from an existing weights file, that file path can be provided in the weights argument, but be aware that these weigths will be overwritten.

Finally, training data paths must be passed to the model in the form of a features file and a labels file. Sample training data can be found in the `data` folder with `X.csv` being the features data and `Y.csv` being the labels data.

A complete training command is as follows:
`./DNN train ../config/l3.config ../data/test.weights ../data/X.csv ../data/Y.csv`

### Testing

Testing can be run on the deep neural network model to determine if training has been successful and the model has learned the desired relationships between features and labels. Testing should be done using data that wasn't passed to the model during training, but comes from the same larger dataset that produced the training data.

A complete testing command is as follows:
`./DNN test ../config/l3.config ../data/test_5-18.weights ../data/X.csv ../data/Y.csv`

### Inferencing

Finally, a trained model can be used for inferencing, which is essentially predicting the label(s) of a given set of feature data. For predicting, only feature data is passed to the model.

A complete prediction command is as follows:
`./DNN pred ../config/l3.config ../data/test_5-18.weights ../data/X.csv`

## Source Code

A bief overview of the project sorce code.

- activation.cpp & activation.h
  - Implements the ReLU and Softmax activation functions of the neural network.
- layer.cpp & layer.h
  - Implements the Dense layer of the neural network.
- loss.cpp & loss.h
  - Implements the Categorical Cross-Entropy loss function of the neural network.
- optimizer.cpp & optimizer.h
  - Implements the Stocastic Gradient Descent optimizer function of the neural network.
- utils.cpp & utils.h
  - Implements various utility functions such as parsing config file, data files, weights file.
- model.cpp & model.h
  - Implements the deep neural network object as well as provides methods for training, testing, and inferencing the model.
- main.cpp
  - This is the entry point of the program and handles all arguments passed to the program.
