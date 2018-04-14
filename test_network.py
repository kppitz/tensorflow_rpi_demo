#File for testing network
# python test_network.py --image tennis_ball_test.jpg

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model("tennis_ball.model")

# classify the input image
(not_tennis_ball, tennis_ball) = model.predict(image)[0]

# build the label
label = "tennis_ball" if ((tennis_ball > not_tennis_ball) and (tennis_ball > 0.7)) else "not_tennis_ball"
probability = tennis_ball if (tennis_ball > not_tennis_ball) and (tennis_ball > 0.7) else not_tennis_ball
label = "{}: {:.2f}%".format(label, probability * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
