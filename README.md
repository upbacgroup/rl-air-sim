# Airsim Curriculum Drone Racing Lab Setup

This repository provides a custom setup for drone racing simulations using AirSim, focusing on training AI models using different reinforcement learning libraries. Follow the steps below to set up and start training your drone models.

## Prerequisites

Before you begin, ensure you have downloaded the AirSim binary files suitable for your operating system. These binaries are necessary to run the simulation environments for drone racing.

## Installation

### Setting Up the Environment

1. **Clone the Repository**:
   - Clone this repository to your local machine using:
     ```bash
     git clone https://github.com/upbacgroup/rl-air-sim.git
     cd rl-air-sim
     ```

2. **Create a Conda Environment**:
   - Set up a conda environment to manage the dependencies:
     ```bash
     conda create -n airsim python=3.10
     conda activate airsim
     pip install -r requirements.txt
     ```

3. **Download AirSim Binaries**:
   - Download the latest version of AirSim binaries from the [official releases page](https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases).

4. **Configure AirSim**:
   - Replace the `settings.json` file in your Documents folder with the `settings.json` file from this repository. This step is crucial as it configures the simulation environment to suit the training requirements.

## Starting the Simulation

To start a new training session:

1. **Launch AirSim**:
   - Open a terminal and run the following command to start the AirSim environment in windowed mode:
     ```bash
     ./ADRL.sh -windowed
     ```

2. **Start Training**:
   - In a new terminal, initiate the training process using one of the following commands based on the RL library you prefer:
     ```bash
     python train_sb3.py  # For training with Stable Baselines 3
     python train_crl.py  # For training with Clean RL
     ```


```
@article{madaan2020airsim,
  title={AirSim Drone Racing Lab},
  author={Madaan, Ratnesh and Gyde, Nicholas and Vemprala, Sai and Brown, Matthew and Nagami, Keiko and Taubner, Tim and Cristofalo, Eric and Scaramuzza, Davide and Schwager, Mac and Kapoor, Ashish},
  journal={arXiv preprint arXiv:2003.05654},
  year={2020}
}
```