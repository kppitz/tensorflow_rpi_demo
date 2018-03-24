# tensorflow_rpi_demo
Autonomy demo for tensorflow on rpi3

To train neural network:
python train_network.py --dataset images --model tennis_ball.model

To test network with picture:
python test_network.py --model tennis_ball.model --image examples/tennis_ball_test.jpg

To run tennis ball recognition video feed:
python object_detection.py
