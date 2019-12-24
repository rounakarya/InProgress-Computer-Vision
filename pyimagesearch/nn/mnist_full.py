# import the necessary packages
from pyimagesearch.nn import neuralnetwork as net
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from mlxtend.data import loadlocal_mnist
import numpy as np

# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 28 x 28 = 784-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")

trainX, trainY = loadlocal_mnist(
        images_path='./MNIST Data/train-images.idx3-ubyte',
        labels_path='./MNIST Data/train-labels.idx1-ubyte')

testX, testY = loadlocal_mnist(
        images_path='./MNIST Data/t10k-images.idx3-ubyte',
        labels_path='./MNIST Data/t10k-labels.idx1-ubyte')

trainX = trainX.astype("float")
testX = testX.astype("float")
trainX = (trainX - trainX.min()) / (trainX.max() - trainX.min())
testX = (testX - testX.min()) / (testX.max() - testX.min())

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print("[INFO] training network...")
nn = net.NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))