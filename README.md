# motor-babbling
The objective of this project is to build a RL system that learns how to properly actuate a robot and move it's end effector into specific target positions

## Project Structure
```
motor-babbling/
├── controller/
│   └── environment.py          # MuJoCo gym environment (2-DOF reaching task)
├── model/
│   └── dqn_agent.py           # DQN agent implementation
├── examples/
│   └── Q_learning.py          # Training and evaluation script
├── data/
│   ├── checkpoints/           # Saved model checkpoints
│   └── plots/                 # Training plots
```

## Installation
```bash
conda create -n motor-babbling python=3.12
conda activate motor-babbling
pip install gymnasium mujoco torch numpy matplotlib
```

## Usage

### Training
```bash
# Basic training (1000 episodes)
python examples/Q_learning.py train

# Training with GUI visualization
python examples/Q_learning.py train --gui

# Training with random targets (better generalization)
python examples/Q_learning.py train --random-targets --episodes 10000

# Resume from checkpoint
python examples/Q_learning.py train --resume
```

### Evaluation
```bash
# Evaluate trained model with GUI
python examples/Q_learning.py eval --gui --random-targets

# Evaluate specific checkpoint
python examples/Q_learning.py eval --load-path data/checkpoints/my_model.pth --gui
```

## Key Parameters

- `--episodes`: Number of training episodes (default: 1000)
- `--max-steps`: Maximum steps per episode (default: 500)
- `--lr`: Learning rate (default: 1e-3)
- `--epsilon-decay`: Epsilon decay rate (default: 0.995)
  - Faster decay (0.995): Quick convergence, less exploration
  - Slower decay (0.9995): More exploration, better generalization (needs more episodes)
- `--random-targets`: Randomize target position each episode

## Output Files

Training generates:
- `data/checkpoints/dqn_robot.pth` - Model checkpoint
- `data/plots/dqn_robot_training.png` - Training plots (150 DPI)
- `data/plots/dqn_robot_training_hires.png` - High-res plots (300 DPI)
- `data/plots/dqn_robot_training_data.npz` - Raw training data


