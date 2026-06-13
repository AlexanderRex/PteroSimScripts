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

Virtual environment is required for smooth usage. Create one **in this repo** and activate it before every session.

**Windows — PowerShell**

```powershell
cd your\path\to\PteroSimScripts
python -m venv pterosim-venv
.\pterosim-venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS — bash**

```bash
cd your/path/to/PteroSimScripts
python3 -m venv pterosim-venv
source pterosim-venv/bin/activate
```

Deactivate when finished:

```bash
deactivate
```

To **remove** the virtual environment (e.g. recreate from scratch or reclaim disk space), run **`deactivate` first** if the venv is still active (your prompt shows `(pterosim-venv)`). Then delete the `pterosim-venv` folder:

**Windows — PowerShell**

```powershell
cd your\path\to\PteroSimScripts
Remove-Item -Recurse -Force pterosim-venv
```

**Linux / macOS — bash**

```bash
cd your/path/to/PteroSimScripts
rm -rf pterosim-venv
```

---

## Install the `pterosim` package

The Python package ships **inside the unpacked simulator archive**:

**Windows — PowerShell**

```powershell
cd your\path\to\Pterosim\Plugins\PteroSimScripting\SDK\python
python -m pip install -U pip
python -m pip install -e .
```

**Linux / macOS — bash**

```bash
cd your/path/to/Pterosim/Plugins/PteroSimScripting/SDK/python
python3 -m pip install -U pip
python3 -m pip install -e .
python -m pip install numpy
```

## Get and run PteroSim

1. Open **[PteroSim](https://github.com/PteroLabsAI/PteroSim-UAV-Simulator/releases/tag/v0.1.0)**.
2. Download the asset for your OS and unpack it.
3. **Windows:** from the unpacked folder, run `PteroSim.exe`. **Linux:** run `PteroSim.sh`.
4. Start a play session in the shipped build so the simulator is running.

You need a **running sim** so the gRPC server in `PteroSimScripting` is listening.

---

## Connect from Python

With the virtual environment **activated** and PteroSim **running**:

```python
from pterosim import PteroSim

sim = PteroSim("localhost:10010")

```

Remote machine: use `"192.168.1.10:10010"` (host running the simulator) instead of `localhost`, and allow TCP on that port through the firewall.
