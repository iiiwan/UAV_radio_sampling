# UAV-enabled Adaptive Radio Sampling for High-quality Radio Environment Map Construction in Unknown Environments

This repository contains the source code for the paper "UAV-enabled Adaptive Radio Sampling for High-quality Radio Environment Map Construction in Unknown Environments".

## Overview

Radio environment maps (REMs) integrate multi-domain environmental data from real-world wireless scenarios, of which the high-quality construction is crucial for optimizing wireless communication and ensuring efficient spectrum utilization. Owing to the high mobility and flexibility, UAV-enabled radio sampling is highly effective for REM construction by informative path planning in information-rich positions. However, existing UAV sampling strategies rely solely on Gaussian process (GP) uncertainty reduction, which induces structural inaccuracies in REMs, particularly in environments with unknown obstacles. To address this issue, we first introduce a dual-information metric that combines uncertainty reduction with signal variation magnitude. Then we formulate an adaptive path planning problem for radio sampling under the energy constraint. Considering the NP-hardness of this problem, we propose a deep reinforcement learning-based adaptive sampling for UAV, termed DRL-ASU. To be specific, we propose an obstacle-aware probabilistic roadmap (PRM) generation mechanism that transforms the initial problem into a sequential graph-based decision problem. On this basis, we design an attention-based policy model for efficient resolution, optimized via multi-environment proximal policy optimization (PPO) training, which enables robust cross-scenario adaptability in diverse radio environments.

### Key Contributions

- To address the deficiencies in REM construction caused by relying solely on uncertainty-based sampling, we first introduce the dual-information metric that jointly considers both uncertainty reduction and the magnitude of signal variation.
- We propose an obstacle-aware PRM generation mechanism to encapsulate connectivity information between the current location and subsequent positions. On this basis, the initial problem is transformed into a sequential graph-based decision problem, thereby reducing computational complexity of continuous space search. 
- Our DRL-ASU scheme employs an attention-based policy to generate the optimal next sampling position, optimized via multi-environment PPO training, which enables robust cross-scenario adaptability in diverse radio environments. 

## Code Structure

```
├── attention_net.py       # Neural network implementation
├── env.py                 # Simulation environment for REM construction
├── gp_ipp.py              # Gaussian Process with information-theoretic planning
├── driver.py              # Training driver script
├── runner.py              # Training runner script
├── worker.py              # Training worker implementation
├── main.py                # Main entry point
├── parameters.py          # Training parameters configuration
├── eval/                  # Evaluation scripts
│   ├── test_driver.py     # Test driver script
│   ├── test_worker.py     # Test worker implementation
│   └── test_parameters.py # Test parameters configuration
└── classes/               # Utility classes
    ├── Gaussian2D.py      # 2D Gaussian distribution
    ├── Graph.py           # Graph structure for path planning
    ├── Utils.py           # Utility functions
    ├── create_map.py      # Map generation
    ├── graph_generater.py # Graph generation
    └── obstacle.py        # Obstacle handling
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- SciPy
- matplotlib
- torch_geometric
- imageio
- scikit-image

### Training
python driver.py
```

### Evaluation
cd eval
python test_driver.py
```

### Configuration

Training parameters can be adjusted in `parameters.py`, including:
- Network hyperparameters
- Training settings
- Environment configuration

## License

This project is licensed under the MIT License.

## Note

This is a preliminary version of the code. The complete code, including detailed documentation and additional experimental results, will be made publicly available upon acceptance of the paper.
```
