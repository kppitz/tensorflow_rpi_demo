#USAGE
# python net.py --dataset images

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from imutils import paths
import argparse
import cv2
import random
import os

dimensions = 224 # both height and width of images
classes = 2 # current classes: tennis_ball and not_tennis_ball
data = []
labels = []

#read in images
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["dataset"])))

random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (dimensions, dimensions))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "tennis_ball" else 0
	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=classes)
testY = to_categorical(testY, num_classes=classes)

print("Compiling model")
model = Sequential()
# input: 224x224 images with 3 channels -> (224, 224, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dimensions, dimensions, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print("Training model")
model.fit(trainX, trainY, batch_size=32, epochs=10)
score = model.evaluate(testX, testY, batch_size=32)

model.save("tennis_ball.model")
