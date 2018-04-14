# tensorflow_rpi_demo
Autonomy demo for tensorflow

To train neural network:
python net.py --dataset images

To test network with a picture:
python test_network.py --model tennis_ball.model --image tennis_ball_test.jpg

To run tennis ball recognition video feed:
python object_detection.py

To install necessary packages in conda environment:
conda install keras
conda install opencv
conda install matplotlib
conda install scikit-learn
conda install -c mlgill imutils
pip install imutils
python -m pip install --upgrade pip
