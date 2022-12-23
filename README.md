#6 DOF Arm Chess Playing Robot

This repository contains the code and documentation for a 6 DOF arm chess playing robot that uses multi-agent reinforcement learning (RL) and an Xbox Kinect RGB-D camera for perception.
##Overview

The robot consists of a 6 DOF arm mounted on a base, with an Xbox Kinect RGB-D camera positioned facing the arm and chess board. The robot uses the camera to perceive the chess board and the pieces, and uses multi-agent RL to learn how to play chess.
Dependencies

In order to run the code in this repository, you will need the following dependencies:

    Python 3.6 or higher
    NumPy
    PyGame
    OpenAI Gym
    TensorFlow or PyTorch (for training the RL agents)

##Setup

To set up the robot, follow these steps:

    Install the dependencies listed above.
    Clone this repository onto your local machine.
    Connect the Xbox Kinect camera to your computer.
    Connect the 6 DOF arm to your computer via USB.
    Calibrate the arm and camera using the provided calibration scripts.

##Running the code

To run the code, follow these steps:

    Open a terminal and navigate to the root directory of this repository.
    Run the command python main.py.

Training the RL agents

To train the RL agents, follow these steps:

    Open a terminal and navigate to the agents directory.
    Run the command python train.py.

##Credits

This project was developed by Jhoel Witter.
