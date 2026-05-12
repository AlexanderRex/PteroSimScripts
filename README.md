# PteroSim Scripts

This repository holds **Python scripts, examples, and automation** for PteroSim: RL loops, tooling, and utilities that use the **gRPC Python SDK** (`pterosim`).

Get the simulator from **[PteroSim v0.1.0 open beta (GitHub Release)](https://github.com/PteroLabsAI/PteroSim-UAV-Simulator/releases/tag/v0.1.0)** — download the build for your OS and read the release notes there.

---

## Table of contents

- [Virtual environment (required)](#virtual-environment-required)
- [Install the `pterosim` package](#install-the-pterosim-package)
- [Get and run PteroSim](#get-and-run-pterosim)
- [Connect from Python](#connect-from-python)
- [Python SDK layout](#python-sdk-layout)
- [Change the gRPC port](#change-the-grpc-port)
- [Git hooks (this repo)](#git-hooks-this-repo)

---

## Virtual environment (required)

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

**Windows — Command Prompt**

```cmd
cd %USERPROFILE%\Documents\PteroSimScripts
python -m venv .venv
.\.venv\Scripts\activate.bat
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

Point `pip` at the folder that contains `pyproject.toml` for package `pterosim`. In the **v0.1.0 open-beta** archives that path is always:

`<unpacked_release>/PteroSim/Plugins/PteroSimScripting/SDK/python`

Set `PTEROSIM_INSTALL` to the **`PteroSim` directory inside the unpacked build** (the one that sits next to `Engine` at the top level and contains `Binaries`, `Content`, `Plugins`).

**Windows — PowerShell** (default unpack under Downloads)

```powershell
$env:PTEROSIM_INSTALL = "$env:USERPROFILE\Downloads\PteroSim-v0.1.0-Windows\PteroSim"
python -m pip install -U pip
python -m pip install -e "$env:PTEROSIM_INSTALL\Plugins\PteroSimScripting\SDK\python"
```

**Linux / macOS — bash** (same hierarchy; archive name matches the Linux asset, e.g. `PteroSim-v0.1.0-Linux`)

```bash
export PTEROSIM_INSTALL="$HOME/Downloads/PteroSim-v0.1.0-Linux/PteroSim"
python3 -m pip install -U pip
python3 -m pip install -e "$PTEROSIM_INSTALL/Plugins/PteroSimScripting/SDK/python"
```

If you copied only the SDK tree elsewhere:

**Windows — PowerShell**

```powershell
cd C:\Path\To\PteroSimScripting\SDK\python
python -m pip install -e .
```

**Linux / macOS — bash**

```bash
cd ~/path/to/PteroSimScripting/SDK/python
python3 -m pip install -e .
```

Optional extras for working on the SDK or running its tests (only if that tree is present):

**Windows — PowerShell**

```powershell
python -m pip install -e "$env:PTEROSIM_INSTALL\Plugins\PteroSimScripting\SDK\python[dev]"
```

**Linux / macOS — bash**

```bash
python3 -m pip install -e "$PTEROSIM_INSTALL/Plugins/PteroSimScripting/SDK/python[dev]"
```

---

## Get and run PteroSim

1. Open **[PteroSim v0.1.0-open-beta](https://github.com/PteroLabsAI/PteroSim-UAV-Simulator/releases/tag/v0.1.0)**.
2. Download the **asset** for your OS and unpack it. You get a root folder (e.g. `PteroSim-v0.1.0-Windows`) containing `Engine`, `PteroSim.exe` (Windows) or the Linux launcher/binary from that asset, plus a **`PteroSim`** subfolder (game content: `Binaries`, `Content`, `Plugins`).
3. **Windows:** from the unpacked root, run `PteroSim.exe`. **Linux:** run the launcher or binary supplied in that root (same folder layout; see the release asset).
4. Start a play session in the shipped build so the simulator is running.

You need a **running game** so the gRPC server in `PteroSimScripting` is listening (default `localhost:10010`).

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

---

## Python SDK layout (v0.1.0 open-beta)

Checked on **`PteroSim-v0.1.0-Windows`**; the Linux archive matches this layout.

```text
<unpacked_root>/
  PteroSim.exe                 # Windows — launcher at shipping root
  Engine/
  PteroSim/
    Binaries/
    Content/
    Plugins/
      PteroSimScripting/
        SDK/
          python/              # pip install -e this directory (pyproject.toml here)
            pterosim/
  FileOpenOrder/
  Manifest_*.txt
```

Use `PTEROSIM_INSTALL=<unpacked_root>/PteroSim` in the [install](#install-the-pterosim-package) commands above.

---

## Change the gRPC port

Default port is **10010**. In Unreal, set console variable **`pterosim.GrpcPort`** before or at startup if you need another port, then pass the same host:port to `PteroSim("host:port")`.

---

## Git hooks (this repo)

Enable pre-commit linting for **PteroSimScripts** itself (Ruff):

**Windows — PowerShell**

```powershell
cd $env:USERPROFILE\Documents\PteroSimScripts
git config core.hooksPath .githooks
```

**Linux / macOS — bash**

```bash
cd ~/Documents/PteroSimScripts
git config core.hooksPath .githooks
```
