from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

#path to trained keras model
MODEL_PATH = "tennis_ball.model"

#total consecutive frames tennis ball recognized
TOTAL_CONSEC = 0
#how many frames until confirming tennis tennis_ball
TOTAL_THRESH = 50
#tennis ball spotted
TENNIS_BALL = False

#load trained model
print("Loading model")
model = load_model(MODEL_PATH)

#Loading video feed
print("Starting video stream")
vs = VideoStream(0).start()
time.sleep(2.0)

while True:
    #read and resize each frame
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #prepare image to be classified
    image = cv2.resize(frame, (28, 28))
    image = image.astype("float")/255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    #classify image
    (not_tennis_ball, tennis_ball) = model.predict(image)[0]
    label = "not_tennis_ball"
    probability = not_tennis_ball

    #if tennis ball is detected
    if (tennis_ball > not_tennis_ball) and (tennis_ball > 0.5):
        label = "tennis_ball"
        probability = tennis_ball
        TOTAL_CONSEC += 1

    #check if spotted in enough consecutive frames
    if not TENNIS_BALL and TOTAL_CONSEC >= TOTAL_THRESH:
        TENNIS_BALL = True

        #processing, math, calling other functions here

    else:
        TOTAL_CONSEC = 0
        TENNIS_BALL = False

    #prints likelihood of being a tennis ball
    label = "{}: {:.2f}%".format(label, probability * 100)
    frame = cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2)

    #output frame
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("Closing")
cv2.destroyAllWindows()
vs.stop()
