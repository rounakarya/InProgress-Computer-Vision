# import the necessary packages
from pyimagesearch.nn import neuralnetwork as net
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import cv2
import numpy as np

# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()

    # Try displaying an image of the MNIST dataset
    #img = digits.images[0]
    # cv2.imwrite("image.jpg", img)
    #cv2.imshow("image", img)
    # cv2.waitKey()

#imag = cv2.imread("image3.bmp")
#gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
#print(imag)

data = digits.data.astype("float")
# normalizing the data between 0-1 using min-max normalization technique
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

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