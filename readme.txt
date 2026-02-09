A reinforcement learning project using Stable-Baselines3 and Highway-env to train an agent for intersection navigation with PPO algorithm.
Project Structure
RL3/
├── models/                    # Trained model files
├── train.py                  # Training script
├── test.py                   # Testing script
└── README.md                 # Project documentation
Requirements
Core Dependencies
pip install gymnasium highway-env stable-baselines3 torch
Optional Dependencies (GPU Acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Quick Start
1. Train the Model
Run the training script:
python train.py
Training Parameters:
Algorithm: PPO (Proximal Policy Optimization)
Training steps: 2050 steps
Environment: intersection-v0 (intersection scenario)
Output: Model saved to models/ppo_intersection_v0.zip
2. Test the Model
Run the testing script:
python test.py
Testing Parameters:
Test episodes: 30 games
Render mode: rgb_array
Policy: Deterministic with some randomness
Code Description
train.py - Training Script
Creates custom intersection environment
Uses PadObservationwrapper to standardize observation shape
Configures PPO algorithm parameters for training
Automatically saves trained model
test.py - Testing Script
Loads pre-trained model
Runs agent in visual environment
Displays real-time rewards and action selections
Calculates total rewards per episode
Environment Configuration
Intersection environment key parameters:
Observation space: Kinematic information of 10 vehicles (position, velocity, direction, etc.)
Action space: Discrete lateral control actions
Reward function: Collision penalty -5.0
Objective: Safely reach the designated destination
Key Features
Custom Observation Wrapper
PadObservationclass ensures consistent observation shape (15, 7)
Handles variable number of vehicles in observation space
Maintains original environment's observation characteristics
Training Optimization
Single environment training setup
Windows multiprocessing support
Progress bar for training monitoring
Automatic directory creation for model storage
Testing and Evaluation
30-episode evaluation protocol
Real-time rendering and action logging
Reward tracking per episode
Deterministic policy with exploration capability
Important Notes
Path Configuration: Modify BASE_DIRand MODEL_PATHaccording to your actual directory structure
Hardware Requirements: Uncomment device="cuda"in training code for GPU acceleration
Training Stability: Recommended to set training steps as multiples of 2048
Rendering Issues: If encountering display problems, try modifying render_modeparameter