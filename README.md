# motor-babbling
The objective of this project is to build a RL system that learns how to properly actuate a robot and move it's end effector into specific target positions

## Project Structure
```
motor-babbling/
├── controller/
│   └── environment.py          # MuJoCo Gymnasium environment (RobotReachingEnv)
├── model/
│   ├── agent.py                # Abstract BaseAgent class
│   ├── dqn_agent.py            # DQN agent (discrete actions)
│   ├── ddpg_agent.py           # DDPG agent (continuous actions)
│   ├── sac_agent.py            # SAC agent (continuous actions)
│   └── replay_buffer.py        # Shared experience replay buffer
├── examples/
│   ├── Q_learning.py           # Main training, evaluation, and play script
│   └── control_model.py        # Manual control via MuJoCo viewer sliders
├── data/
│   ├── checkpoints/            # Saved model checkpoints (.pth)
│   └── plots/                  # Training plots and raw data (.png, .npz)
```

## Installation
```bash
conda create -n motor-babbling python=3.12
conda activate motor-babbling
pip install gymnasium mujoco torch numpy matplotlib
```

## Agents
 
The project uses a plugin style architecture. Agents are loaded dynamically via `importlib` based on the `--agent` flag, so adding a new agent only requires creating a `{name}_agent.py` file in `model/` that inherits from `BaseAgent`.
 
**DQN** -- Discrete action space. Torques are quantized into a fixed number of levels per joint (controlled by `action_quantization`). Uses epsilon greedy exploration.
 
**DDPG** -- Continuous action space. Deterministic policy with Gaussian exploration noise. Actor critic architecture with soft target updates.
 
**SAC** -- Continuous action space. Stochastic policy with automatic entropy tuning. Twin Q networks for stability. Generally the most stable of the three for this task.


## Environment
 
`RobotReachingEnv` is a Gymnasium environment wrapping a MuJoCo simulation of a planar robot arm.
 
**Configuration:**
- `num_links` -- Number of arm links (each 0.5m long, hinge joints on Y axis)
- `continuous` -- Toggle between continuous (`Box`) and discrete (`Discrete`) action spaces
- `action_quantization` -- Number of discrete torque levels per joint (only used when `continuous=False`)
- `max_torque` -- Maximum torque applied through MuJoCo actuators (default 60.0, set via the actuator gear ratio)
 
**Observation space** (dimension = 3 * num_links + 4):
- Joint angles, joint velocities, joint accelerations (one per link each)
- End effector XZ position (2D, planar)
- Target XZ position (2D, planar)

**Reward:** Negative Euclidean distance from end effector to target (`-distance`).

**Termination:** The episode terminates (success) when the end effector stays within 0.05m of the target for 100 consecutive simulation steps. There is no truncation by default; the `--max-steps` flag in the training script handles episode length limits.
 

## Usage
 
All commands are run from the project root. The main script is `examples/Q_learning.py`.
 
### Training

This will begin network training according to the model selected.
 
```bash
# Train SAC for 500 episodes with random targets
python examples/Q_learning.py --agent sac train --episodes 500 --random-targets
 
# Train DQN (discrete actions)
python examples/Q_learning.py --agent dqn train --episodes 1000 --random-targets
 
# Train DDPG
python examples/Q_learning.py --agent ddpg train --episodes 1000 --random-targets
 
# Train with GUI visualization (slower)
python examples/Q_learning.py --agent sac train --episodes 500 --gui
 
# Resume training from a checkpoint
python examples/Q_learning.py --agent sac train --resume --save-path data/checkpoints/sac.pth
 
# Custom save path
python examples/Q_learning.py --agent sac train --save-path data/checkpoints/sac_experiment1.pth
```
 
### Evaluation

This will evaluate the model by checking how many episodes in total it manages to reach.
 
```bash
# Evaluate with GUI and random targets
python examples/Q_learning.py --agent sac eval --gui --eval-episodes 3 --random-targets
 
# Headless evaluation (prints success rate)
python examples/Q_learning.py --agent sac eval --random-targets --eval-episodes 50
```
 
### Play (watch indefinitely)

Allows the user to adjust the position of the target using the arrow keys to see how the robot model dynamically adjusts in real time to a changing target. 

```bash
python examples/Q_learning.py --agent sac play --random-targets
```
 
### Manual Control
 
Allows the user to manually control the robots joint torques. (No model control at all).

```bash
# Open MuJoCo viewer with manual slider control
python examples/control_model.py --num-links 2
```


## Training Parameters
 
### General
 
| Flag | Default | Description |
|---|---|---|
| `--agent` | (required) | Agent type: `dqn`, `ddpg`, `sac` |
| `--episodes` | 500 | Number of training episodes |
| `--max-steps` | 500 | Max simulation steps per episode |
| `--gamma` | 0.99 | Discount factor |
| `--buffer-size` | 50000 | Replay buffer capacity |
| `--batch-size` | 32 | Training batch size |
| `--random-targets` | off | Randomize target each episode |
| `--gui` | off | Show MuJoCo viewer during training |
| `--resume` | off | Resume from existing checkpoint |
| `--save-path` | `data/checkpoints/agent.pth` | Checkpoint save location |
| `--save-freq` | 100 | Save checkpoint every N episodes |
| `--print-freq` | 10 | Print metrics every N episodes |
 
### DQN Specific
 
| Flag | Default | Description |
|---|---|---|
| `--lr` | 1e-3 | Learning rate |
| `--epsilon-start` | 1.0 | Initial exploration rate |
| `--epsilon-end` | 0.01 | Minimum exploration rate |
| `--epsilon-decay` | 0.99999 | Per step epsilon decay multiplier |
| `--target-update-freq` | 100 | Steps between target network updates |
 
### DDPG/SAC Specific
 
| Flag | Default | Description |
|---|---|---|
| `--actor-lr` | 1e-4 | Actor learning rate |
| `--critic-lr` | 1e-3 | Critic learning rate |
| `--tau` | 0.005 | Soft target update coefficient |
| `--noise-std` | 0.1 | Exploration noise std (DDPG only) |
 
Note: `num_links`, `action_quantization`, and `decay_rate` are currently set as module level constants at the top of `Q_learning.py` rather than as CLI flags.

## Output Files
 
Training produces four files per run, named based on the `--save-path` basename:
 
- `data/checkpoints/<name>.pth` -- Model checkpoint (loadable for eval/play/resume)
- `data/plots/<name>_training.png` -- Training plots at 150 DPI
- `data/plots/<name>_training_hires.png` -- Training plots at 300 DPI
- `data/plots/<name>_training_data.npz` -- Raw training data (rewards, losses) for custom analysis
 
DQN plots show reward and loss curves. DDPG and SAC plots show reward, critic loss, and actor loss curves. All include 50 episode moving averages.
 
## Adding a New Agent
 
1. Create `model/{name}_agent.py` with a class `{NAME}Agent` inheriting from `BaseAgent`
2. Implement all abstract methods: `select_action`, `store_transition`, `train_step`, `save`, `load`
3. If the agent uses continuous actions, add its name to the `continuous_agents` list in `requires_continuous_actions()` in `Q_learning.py`
4. If the agent needs custom constructor parameters beyond the common set, add a branch in `get_agent()`
 
The dynamic import system will handle the rest: `--agent myagent` loads `model.myagent_agent.MYAGENTAgent`.

## Acknowledgements  

AI was used to aid in code development, commentation and editing parts of the report



