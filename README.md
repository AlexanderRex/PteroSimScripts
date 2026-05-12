# PteroSim Scripts

This repository holds **Python scripts, examples, and automation** for PteroSim: RL loops, tooling, and utilities that use the **gRPC Python SDK** (`pterosim`).

Get the simulator from **[PteroSim v0.1.0 open beta (GitHub Release)](https://github.com/PteroLabsAI/PteroSim-UAV-Simulator/releases/tag/v0.1.0)** — download the build for your OS and read the release notes there.

---

## Table of contents

- [Virtual environment](#virtual-environment-required)
- [Install the `pterosim` package](#install-the-pterosim-package)
- [Get and run PteroSim](#get-and-run-pterosim)
- [Connect from Python](#connect-from-python)

---

## Virtual environment

Skipping a venv means **mixed `pip` installs, broken `grpcio` / `protobuf` versions, and permission pain on system Python**. Create one **in this repo** and activate it before every session.

**Windows — PowerShell**

```powershell
cd $env:USERPROFILE\Documents\PteroSimScripts
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS — bash**

```bash
cd ~/Documents/PteroSimScripts
python3 -m venv .venv
source .venv/bin/activate
```

Deactivate when finished:

```bash
deactivate
```

---

## Install the `pterosim` package

The Python package ships **inside the unpacked archive**. From the archive root, do:



**Windows — PowerShell**

```powershell
cd PteroSim\Plugins\PteroSimScripting\SDK\python
python -m pip install -U pip
python -m pip install -e .
```

**Linux / macOS — bash**

```bash
cd ~/PteroSim/Plugins/PteroSimScripting/SDK/python
python3 -m pip install -U pip
python3 -m pip install -e .
```

## Get and run PteroSim

1. Open **[PteroSim](https://github.com/PteroLabsAI/PteroSim-UAV-Simulator/releases/tag/v0.1.0)**.
2. Download the asset for your OS and unpack it.
3. **Windows:** from the unpacked root, run `PteroSim.exe`. **Linux:** run `PteroSim.sh`.
4. Start a play session in the shipped build so the simulator is running.

You need a **running sim** so the gRPC server in `PteroSimScripting` is listening.

---

## Connect from Python

With the venv **activated** and PteroSim **running**:

```python
from pterosim import PteroSim

sim = PteroSim("localhost:10010")
# e.g. sim.start(), sim.spawn("F450", x=0, y=0, z=200), ...
sim.close()
```

Remote machine: use `"192.168.1.10:10010"` (host running the simulator) instead of `localhost`, and allow TCP on that port through the firewall.
