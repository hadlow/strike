import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import struct as st

import layer
import cost
import optimizer

def process_image_file(image_file):
    image_file.seek(0)
    magic = st.unpack('>4B', image_file.read(4))

    n_images = st.unpack('>I', image_file.read(4))[0]
    n_rows = st.unpack('>I', image_file.read(4))[0]
    n_columns = st.unpack('>I', image_file.read(4))[0]
    n_bytes = n_images * n_rows * n_columns

    images = np.zeros((n_images, n_rows * n_columns))
    images = np.asarray(st.unpack('>' + 'B' * n_bytes, image_file.read(n_bytes))).reshape((n_images, n_rows * n_columns))

    return images

def process_label_file(label_file):
    label_file.seek(0)
    magic = st.unpack('>4B', label_file.read(4))

    n_labels = st.unpack('>I', label_file.read(4))[0]

    labels = np.zeros((n_labels))
    labels = np.asarray(st.unpack('>' + 'B' * n_labels, label_file.read(n_labels)))

    targets = np.array([labels]).reshape(-1)

    one_hot_labels = np.eye(10)[targets]

    return one_hot_labels

def dataset():
    home = os.path.expanduser('~') + '/Datasets/'

    test_images = open(home + 't10k-images-idx3-ubyte', 'rb')
    test_labels = open(home + 't10k-labels-idx1-ubyte', 'rb')
    train_images = open(home + 'train-images-idx3-ubyte', 'rb')
    train_labels = open(home + 'train-labels-idx1-ubyte', 'rb')
    
    train_images = process_image_file(train_images)
    test_images = process_image_file(test_images)
    train_labels = process_label_file(train_labels)
    test_labels = process_label_file(test_labels)
    
    return ((train_images, test_images), (train_labels, test_labels))

# Get MNIST training data
((train_images, test_images), (train_labels, test_labels)) = dataset()
train_images, test_images = train_images / 255.0, test_images / 255.0

n_training_examples = 60000
n_inputs = 784
n_samples = 32
n_epochs = 5
n_batches = math.floor(n_training_examples / n_samples)
learning_rate = 4

layer1 = layer.Dense(n_inputs = n_inputs, n_neurons = 128)
activation1 = layer.ReLU()

layer2 = layer.Dense(n_inputs = 128, n_neurons = 10)
activation2 = layer.Sigmoid()

def forward(inputs):
    x = layer1.forward(inputs)
    x = activation1.forward(x)

    x = layer2.forward(x)
    activation2.forward(x)

    return activation2

for j in range(n_epochs):
    shuffle_index = np.random.permutation(n_training_examples)

    for batch_start in shuffle_index:
        batch_end = batch_start + n_samples

        batch_x = train_images[batch_start:batch_end]
        batch_y = train_labels[batch_start:batch_end]

        output = forward(batch_x)
        (layer1_dw, layer1_db, layer2_dw, layer2_db) = backward(batch_x, batch_y, output)

        layer1.weights = layer1.weights - learning_rate * layer1_dw
        layer1.biases = layer1.biases - learning_rate * layer1_db
        layer2.weights = layer2.weights - learning_rate * layer2_dw
        layer2.biases = layer2.biases - learning_rate * layer2_db

    train_output = forward(train_images)
    train_loss = batch_loss(train_labels, train_output)

    test_output = forward(test_images)
    test_loss = batch_loss(test_labels, test_output)

    print("€poch {}, training loss: {}, test loss: {}", format(j + 1, train_loss, test_loss))
