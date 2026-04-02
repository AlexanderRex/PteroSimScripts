# PteroSim Python Examples

Python scripts and examples for [PteroSim](https://github.com/PteroLabsAI/PteroSim-UAV-Simulator) UAV simulator.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Virtual Environment](#virtual-environment)

## Prerequisites

- Python 3.9+

## Setup

```bash
python -m venv <env_name>
source <env_name>/bin/activate       # Linux/macOS
.\<env_name>\Scripts\Activate.ps1    # Windows

pip install -e /path/to/PteroSim/Plugins/PteroSimScripting/SDK/python
```

Run an example:

```bash
python examples/hello_drone.py
```

## Virtual Environment

| Action | Command |
|--------|---------|
| Create | `python -m venv <env_name>` |
| Activate (Linux/macOS) | `source <env_name>/bin/activate` |
| Activate (Windows) | `.\<env_name>\Scripts\Activate.ps1` |
| Deactivate | `deactivate` |
| Remove (Linux/macOS) | `rm -rf <env_name>` |
| Remove (Windows) | `rmdir /s /q <env_name>` |
