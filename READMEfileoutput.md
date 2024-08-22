# Warehouse Agent Training

This repository contains the implementation of a gym-based environment for warehouse agents and scripts for training these agents using different reinforcement learning algorithms.

## Directory Structure

- **Data/**: Directory for storing training data.
- **robotic_warehouse/**: Contains the necessary files to create the `warehouse.py` environment, a gym-based environment for reinforcement learning.
- **Plots/**: Directory for storing generated plots during training.

## Training Scripts

### 1. **train_sep_maddpg.py**
   - This script is used to train warehouse agents using the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm.
   - The MADDPG algorithm is defined in `maddpg.py`.

### 2. **train_q_learning.py**
   - This script is used to train warehouse agents using the DQN (Deep Q-Network) algorithm.

## Requirements

To install the necessary libraries, please run:

```bash
pip install -r requirements.txt
