# OptiMat Alloys — Setup Guide (deep-dive appendix)

> **Start with the [README](../README.md#-quick-start--docker-recommended)** — it is the canonical install instruction (Docker quick start + source install).
> This guide is the companion appendix: detailed Windows/Docker Desktop walkthroughs, GPU enablement, WSL2 setup, data-volume locations, verification, and troubleshooting.

---

## Guide Map

| Path | Covers | GPU Support |
|------|--------|-------------|
| **[A. Docker Desktop](#path-a-docker-desktop-windows-clickable-install)** | Windows/macOS Docker details: GPU passthrough, compose walkthrough, volumes | ✅ (with NVIDIA driver) |
| **[B. WSL2 + conda](#path-b-wsl2--conda-windows-developers)** | WSL2 details for source installs on Windows | ✅ Native |
| **[C. Native Linux + conda](#path-c-native-linux--conda)** | Source-install details for Linux | ✅ Native |

After installing, all three paths share the same:
- **[First-Time Configuration](#first-time-configuration)** (API keys, Ollama)
- **[Launching the App](#launching-the-app)**
- **[Verification](#verifying-your-setup)**
- **[Troubleshooting](#troubleshooting)**

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10 (21H2+), Windows 11, or Linux (Ubuntu 20.04+) | Windows 11 or Ubuntu 22.04 |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 20 GB free | 30 GB+ |
| **GPU** | Optional — CPU fallback works | NVIDIA GPU with 12 GB+ VRAM |
| **Python** (Paths B/C only) | 3.10–3.12 (installed via conda) | 3.11 |

---

## Path A: Docker Desktop (Windows, clickable install)

A click-through install. No programming experience required.

### A.1. Install Docker Desktop

1. Go to https://www.docker.com/products/docker-desktop/
2. Click **"Download for Windows"**
3. Run the installer and follow the prompts
4. Restart your computer when asked
5. Open Docker Desktop — it should show a green "Running" status in the bottom-left

> **macOS users:** follow Docker's official installer guide at https://docs.docker.com/desktop/setup/install/mac-install/ (it covers both Apple Silicon and Intel). The rest of Path A applies as-is — except A.2, which does not apply to macOS (see the macOS callout in A.2 for why).

### A.2. Enable NVIDIA GPU Access (required for GPU users)

> **CPU users:** skip this section. The steps below only matter if you want the container to see your NVIDIA GPU.
>
> **macOS users:** Docker Desktop on macOS cannot expose any GPU to containers — this is architectural, not a config setting. Containers run inside a hidden Linux VM, the `--gpus` / `driver: nvidia` flow is NVIDIA-Container-Toolkit-only (Linux + Windows-WSL2), and Apple Silicon GPUs use Metal, which has no Docker equivalent. Editing `docker-compose-gpu.yml` will not help. Skip to A.3 and use `docker-compose-cpu.yml`. If you need native Apple Silicon ML acceleration (PyTorch MPS / MLX), run it outside Docker — use Path B/C adapted for macOS.

Docker Desktop GPU passthrough is **Windows + WSL2 backend only**. If you have an NVIDIA GPU on Windows and want the container to use it, **all four steps below are required** — the `docker-compose-gpu.yml` flow in A.3 relies on them and will fail otherwise.

1. **Install WSL2.** Open an **Administrator** PowerShell and run `wsl --install`, then reboot when prompted. If WSL is already installed, run `wsl --update` instead — older kernels lack the GPU paravirtualization support Docker needs.
2. **Install a WSL2-aware NVIDIA driver.** Download the latest Game Ready or Studio driver from https://www.nvidia.com/Download/index.aspx. Install it on **Windows** (not inside WSL) — the same driver covers both.
3. **Enable the WSL2 backend in Docker Desktop.** Open Docker Desktop → **Settings** → **General** and check **"Use the WSL 2 based engine"**, then click **Apply & restart**. The legacy Hyper-V backend cannot expose GPUs.
4. **Verify GPU passthrough.** In PowerShell, run:
   ```powershell
   docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
   ```
   You should see an n-body benchmark print GFLOP/s numbers and exit cleanly. If you get an error like `could not select device driver "" with capabilities: [[gpu]]`, one of steps 1–3 is incomplete — fix that before continuing, otherwise the compose command in A.3 will not see your GPU either.

### A.3. Get the App

#### Option 1: Docker Desktop GUI (No GPU)

> ⚠️ This path **does not enable GPU access** — the Docker Desktop "Run" dialog has no NVIDIA toggle. Use Option 2 below if you have an NVIDIA GPU and want to use it.
>
> ℹ️ Docker Desktop's search bar only finds Docker Hub images. The official OptiMat Alloys image lives on GitHub Container Registry (GHCR), so it must be pulled once from a terminal — after that, everything works from the GUI.

1. Open **Docker Desktop** and wait for the green "Running" status
2. Open a terminal (PowerShell on Windows — see Option 2, Step 1 if unsure) and run:
   ```powershell
   docker pull ghcr.io/optimat-chat/optimat-alloys:latest
   ```
   This downloads the image (~12 GB) — give it a few minutes.
3. In Docker Desktop, go to the **Images** tab
4. Find `ghcr.io/optimat-chat/optimat-alloys` and click **Run**
5. In the run dialog, set **Port** to `8000` and click **Run**
6. Open your browser and go to **http://localhost:8000**

#### Option 2: Docker Compose via PowerShell (Required for GPU)

> 🔥 **GPU users — read this.** The Docker Desktop GUI run flow in **Option 1 cannot enable GPU access** (the run dialog has no NVIDIA toggle). If you want your NVIDIA GPU to be detected by the container, you **must** use this Option 2 / Compose flow with `docker-compose-gpu.yml`. Otherwise:
> - **Cloud AI models** (Ollama Cloud, OpenRouter) — still work fine, no GPU needed.
> - **Simulation calculations** (relaxation, elastic, QHA) — still work, but several times slower.
> - **Local AI model** (`gpt-oss:20b` via Ollama Local) — **effectively will not run.** It needs ~12 GB VRAM; on CPU it loads but generates ~1–2 tokens/second, which is unusable for chat. CPU users should stick to cloud AI models.

Also recommended for CPU users who want persistent storage via named volumes. Works on Windows, macOS, and Linux. Pick the file matching your hardware:

| File | When to use |
|------|-------------|
| **`docker-compose-gpu.yml`** | You have an NVIDIA GPU and up-to-date drivers |
| **`docker-compose-cpu.yml`** | No GPU, or you only plan to use cloud AI models |

**Step 1: Open PowerShell**

On Windows, PowerShell comes pre-installed. Open it one of these ways:

- **Easiest:** Press the **Windows key**, type `PowerShell`, and click **Windows PowerShell** in the results.
- **Alternative:** Press **Win + X** (or right-click the Start button), then choose **Terminal** or **Windows PowerShell** from the menu.
- **Windows 11:** Open the **Start menu** → **All apps** → scroll to **Windows PowerShell** (or **Terminal**) → click to open.

You'll see a blue or black window with a `PS C:\Users\YourName>` prompt. That's PowerShell — you can now type the commands below.

**Step 2: Download the compose file**

In PowerShell, create a folder and download the file you need (replace `gpu` with `cpu` if no GPU):

```powershell
mkdir C:\OptiMat-Alloys
cd C:\OptiMat-Alloys
curl.exe -O https://raw.githubusercontent.com/OptiMat-Chat/OptiMat-Alloys/main/docker-compose-gpu.yml
```

> Or copy the file out of a cloned repo. Or open the file's "Raw" view on GitHub and **Save As…** into the folder.

**Step 3: Start the container**

From the same folder in PowerShell:

```powershell
docker compose -f docker-compose-gpu.yml up      # GPU
# or
docker compose -f docker-compose-cpu.yml up      # CPU only
```

The first run pulls the ~12 GB image — give it a few minutes. When you see `Your app is available at http://localhost:8000`, open that URL in your browser.

**Step 4: Stopping and restarting**

```powershell
# Stop (in the running terminal): press Ctrl-C
# Or from another PowerShell window:
docker compose -f docker-compose-gpu.yml down

# Restart later:
docker compose -f docker-compose-gpu.yml up
```

> ℹ️ **Why PowerShell instead of the GUI?** The Docker Desktop GUI's "Run" dialog launches single images and doesn't read compose files at all — there's no way to point it at a `.yml` file and start it. The `docker compose` CLI is the only way to use these files. Once the container is running, it will appear in Docker Desktop's **Containers** tab grouped under the project name (`optimat-alloys`), and you can stop/restart it from the GUI from then on.

➡️ Continue to **[First-Time Configuration](#first-time-configuration)**.

---

## Path B: WSL2 + conda (Windows developers)

OptiMat Alloys requires Linux. On Windows, use WSL2 (Windows Subsystem for Linux 2) for native performance and full CUDA support.

### B.1. Install WSL2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

Restart your computer. Ubuntu will be installed automatically.

### B.2. Open Ubuntu Terminal

Search for "Ubuntu" in the Start menu and open it. Set up your Linux username and password when prompted.

### B.3. Follow the Linux Install Steps

Inside the Ubuntu terminal, follow **[Path C: Native Linux + conda](#path-c-native-linux--conda)** below — the steps are identical from here on.

### B.4. Access the App

After launching, open **http://localhost:8000** in your **Windows** browser. WSL2 automatically forwards the port — no extra configuration needed.

> **Performance tip:** Always clone the repo into the WSL2 native filesystem (`~/OptiMat-Alloys`), **not** the mounted Windows drive (`/mnt/c/...`). The latter is 10–100× slower for file I/O.

---

## Path C: Native Linux + conda

For developers who want full control or need to modify the code.

### C.1. Automated Setup (Recommended)

The setup script handles everything:

```bash
# 1. Clone the repository
git clone https://github.com/OptiMat-Chat/OptiMat-Alloys.git OptiMat-Alloys
cd OptiMat-Alloys

# 2. Run the setup script
bash scripts/setup_linux.sh

# 3. Activate the environment
conda activate optimat-alloys

# 4. Launch the app
chainlit run run_chat.py
```

The setup script will:
- Check for conda installation
- Create the `optimat-alloys` conda environment with Python 3.11
- Install all Python dependencies (PyTorch, ASE, ORB, MACE, Chainlit, etc.)
- Set up CUDA support if a GPU is detected
- Optionally install cuML for large systems (5k+ atoms)

### C.2. Manual Setup

If you prefer step-by-step control:

```bash
# 1. Clone the repository
git clone https://github.com/OptiMat-Chat/OptiMat-Alloys.git OptiMat-Alloys
cd OptiMat-Alloys

# 2. Create conda environment
conda create -n optimat-alloys python=3.11 -y
conda activate optimat-alloys

# 3. Install PyTorch
# For GPU:
pip install torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu124
# For CPU only:
# pip install torch>=2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the app
chainlit run run_chat.py
```

➡️ Continue to **[First-Time Configuration](#first-time-configuration)**.

---

## NequIP Support (Optional)

**Skip this section unless you specifically want to use the NequIP calculators** (`nequip-oam-l`, `nequip-oam-xl`, `nequip-mp-l`). ORB and MACE — the default and most common calculators — work without it.

| Path | Action needed for NequIP |
|------|--------------------------|
| **A. Docker** | ✅ Nothing — the NequIP env is baked into the image and `NEQUIP_ENV_PATH` is pre-set |
| **B. WSL2 + conda** | Run the steps below |
| **C. Native Linux + conda** | Run the steps below (the `setup_linux.sh` script does **not** create this env automatically) |

### Why a separate environment?

NequIP and MACE have an irreconcilable dependency conflict:

- **MACE** pins `e3nn==0.4.4`
- **NequIP** requires `e3nn>=0.5.6`

They cannot live in the same Python environment. The app resolves this by running NequIP in a separate conda environment (`optimat-nequip`) and shipping a worker module (`src/core/nequip_worker.py`) that the main app spawns as a subprocess via `conda run -n optimat-nequip ...` whenever a NequIP calculator is selected.

The dispatcher (`src/core/calculators.py`) and worker code are always present in the repo, but **the subprocess call will fail at runtime if the `optimat-nequip` env doesn't exist** — picking a NequIP calculator in the UI will throw `NequIP calculator '…' requires the optimat-nequip environment`.

### Install (Paths B/C only)

```bash
# Create the NequIP environment
conda create -n optimat-nequip python=3.11 -y
conda activate optimat-nequip

# Install NequIP dependencies
pip install torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install ase>=3.24.0 nequip>=0.6.0 "e3nn>=0.5.6" numpy>=1.26.4 scipy>=1.15.1

# Switch back to the main environment for running the app
conda activate optimat-alloys
```

Once created, the main app finds the env automatically (the worker looks at `NEQUIP_ENV_PATH`, defaulting to `~/miniconda3/envs/optimat-nequip/bin`). No app restart needed beyond the next NequIP selection.

---

## First-Time Configuration

When you first open the app at **http://localhost:8000**, it will prompt for API keys based on which AI model you use.

### API Keys

| Model Type | Key Needed | Where to Get |
|------------|-----------|--------------|
| **Ollama Cloud** (default) | Ollama API Key | https://ollama.com/settings/keys |
| **OpenRouter Free** | OpenRouter API Key | https://openrouter.ai/keys |
| **Ollama Local** | None | Requires `ollama` installed locally (Paths B/C) |

Keys are saved to a `.env` file (not tracked in git) for future sessions.

### Step 1: Ollama API Key

For the default model:

1. Go to https://ollama.com/ and create an account
2. Go to https://ollama.com/settings/keys
3. Click **"Add API Key"**, copy the key
4. Paste it into the app and press Enter

### Step 2: Device Authorization

After entering your API key, the app may ask you to authorize this device:

1. A **clickable link** will appear in the chat
2. Click the link — it opens in your browser
3. Log in to your Ollama account
4. Click **"Connect"** to authorize the device
5. Go back to the app and try your question again

This only needs to be done **once** — authorization persists across restarts.

### Step 3: OpenRouter API Key (Optional)

If you want to use OpenRouter models:

1. Switch to an OpenRouter model in the **Settings** panel (gear icon)
2. The app will ask for an OpenRouter API key
3. Go to https://openrouter.ai/ and click **"Get API Key"**
4. Sign in or create an account, click **"Create"**, and paste the key

> **Tip:** Depositing $10+ on your OpenRouter account unlocks higher rate limits (1,000 requests/day instead of 50). This will **not** be spent on free models — it just removes the rate limit cap.

### Local Ollama (Optional, Paths B/C only)

If you want to run AI models locally (no internet needed for AI):

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model (~12 GB download)
ollama pull gpt-oss:20b
```

Ollama starts automatically — no extra steps needed.

---

## Launching the App

### Path A: Docker Desktop

**If you used Option 2 (Docker Compose):** restart from the same folder in PowerShell:

```powershell
cd C:\OptiMat-Alloys
docker compose -f docker-compose-gpu.yml up    # or -cpu.yml
```

To stop: press Ctrl-C, or `docker compose -f docker-compose-gpu.yml down` from another window.

**If you used Option 1 (Docker Desktop GUI):** Containers tab → find OptiMat Alloys → click **Start** (play icon). To stop: click **Stop** (square icon).

Then open **http://localhost:8000** in your browser.

> ⚠️ **Important:** Click **Stop**, never **Delete** — Delete erases ALL DATA inside the container. Data persists across Stop/Start automatically. If you used Docker Compose (Option 2) with named volumes, your structures and Ollama models also survive container deletion (see below).

#### Where is my data actually stored?

The compose files declare two **named volumes** that Docker manages outside the container:

| Volume | Holds | Mounted at (inside container) |
|--------|-------|-------------------------------|
| `alloy-data` | Your generated structures and SQLite database | `/app/structures` |
| `ollama-models` | Pulled Ollama models (~12 GB if you pull `gpt-oss:20b`) | `/root/.ollama` |

Find them on your host:

- **Docker Desktop GUI:** **Volumes** tab → click `alloy-data` or `ollama-models` → see "Stored on disk" path and browse contents.
- **PowerShell / CLI:**
  ```powershell
  docker volume ls                          # list all volumes
  docker volume inspect alloy-data          # shows the Mountpoint path
  ```

The actual on-disk location depends on your platform:

- **Windows (Docker Desktop, WSL2 backend):** `\\wsl$\docker-desktop-data\data\docker\volumes\alloy-data\_data\` — paste this in File Explorer's address bar.
- **macOS (Docker Desktop):** Inside the Docker Desktop VM disk image — not directly browsable from Finder; use the Docker Desktop **Volumes** tab.
- **Linux (native Docker):** `/var/lib/docker/volumes/<project>_alloy-data/_data/` (typically requires `sudo`).

> Volumes survive `docker compose down` and even container deletion. They are only removed by `docker compose down -v`, `docker volume rm`, or the **Volumes** tab → Delete in Docker Desktop. To back up: copy the folder above, or use `docker run --rm -v alloy-data:/data -v ${PWD}:/backup alpine tar czf /backup/alloy-data.tgz /data`.

### Paths B/C: Conda Environment

```bash
# Using the launch script
bash scripts/launch_optimat_alloys.sh

# Or manually
conda activate optimat-alloys
chainlit run run_chat.py

# Or on a custom port
chainlit run run_chat.py --port 8001
```

---

## Verifying Your Setup

After launching, open **http://localhost:8000** and verify:

### 1. Welcome Screen

You should see:
- Current Settings (model, calculator, supercell size)
- Database Statistics (0 structures for new install)

### 2. Test Structure Generation

Type in the chat:

```
Generate a FCC Cu50Ni50 alloy
```

You should see:
- Structure visualization (atomic model)
- Structural analysis (PTM classification)
- RDF plot
- Formation energy value

### 3. Check GPU (Paths B/C only)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If `True`, calculations will use GPU acceleration. If `False`, CPU fallback is used.

For Path A (Docker), check the container log output for:

```
GPU detected: NVIDIA GeForce RTX 3080 (10240 MiB)
```

If you see `No GPU detected`, see [Troubleshooting](#no-gpu-detected-docker).

---

## Switching Models and Supercell Size

Click the **gear icon** (Settings) near the chat input.

| Model | Type | GPU Required | Description |
|-------|------|-------------|-------------|
| **gpt-oss:120b-cloud** | Ollama Cloud | No | Default. Best reliability, runs remotely |
| **gpt-oss:20b** | Ollama Local | **Yes — 12 GB VRAM required** | Runs on your GPU. Private, no internet needed for AI. Will load on CPU but generates ~1–2 tokens/sec — unusable in practice. |
| **GLM-4.5-Air, GPT-OSS** | OpenRouter Free | No | Free cloud models. May hit rate limits |

### Supercell Size

Click gear → **Default Supercell Size**:

- **Small (48 atoms)** — fastest, good for quick tests
- **Medium (512 atoms)** — balanced (default)
- **Large (2048 atoms)** — most accurate, slowest

### Why GPU Matters

| Feature | CPU | GPU |
|---------|-----|-----|
| AI chat (cloud models) | Works fine | Same |
| AI chat (local models) | Too slow | Fast |
| Structure relaxation | Slow (minutes) | Fast (seconds) |
| Elastic calculations | Slow | Fast |
| QHA calculations | Very slow | Fast |

The app works fully on CPU — just slower for calculations. Cloud AI models don't need GPU at all.

---

## Calculator Options

OptiMat Alloys supports multiple force field calculators (change in Settings → gear icon):

| Calculator | Accuracy | Speed | Notes |
|-----------|----------|-------|-------|
| **ORB v3 Conservative** | High | Fast | Default, recommended |
| **ORB v3 Direct** | High | Fast | Alternative ORB variant |
| **MACE-OMAT Medium** | Very High | Medium | Best for accuracy |
| **MACE-MPA Medium** | Very High | Medium | Alternative MACE |
| **NequIP OAM-L** | High | Slow | Requires separate env |
| **NequIP OAM-XL** | Highest | Slowest | Best accuracy, most compute |

---

## Exporting Your Data

### Download Database

The app shows a **"Download Database"** button on the welcome screen. Click it to download your entire database as a `.db` file.

### Export Individual Structures

When viewing a structure report, the app provides downloadable files:
- **PDF report** — full visual report with charts
- **ZIP file** — contains CIF, POSCAR, CSV data files

---

## Troubleshooting

### Port 8000 Already in Use

```bash
# Paths B/C: kill existing process
pkill -f "chainlit run"
# Or use a different port
chainlit run run_chat.py --port 8001
```

For Path A: stop other apps using port 8000, or change the port mapping in Docker Desktop.

### `failed to connect to the docker API` / `dockerDesktopLinuxEngine` not found (Windows)

```
unable to get image 'ghcr.io/optimat-chat/optimat-alloys:latest': failed to connect to the
docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is
correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine:
The system cannot find the file specified.
```

**Cause:** Docker Desktop is not running. The CLI cannot reach its named-pipe daemon, so no `docker` / `docker compose` command will work yet.

**Fix:**
1. Start **Docker Desktop** from the Windows Start menu.
2. Wait for it to fully initialize — the whale icon in the system tray must stop animating, and Docker Desktop's status should show **"Engine running"** (green) in the bottom-left.
3. Re-run your command, e.g. `docker compose -f docker-compose-gpu.yml up`.

The same error appears on macOS/Linux if the Docker daemon isn't running — the fix is the same: start Docker Desktop (macOS) or `sudo systemctl start docker` (Linux).

### "No GPU Detected" (Docker)

**Cause:** Docker is not configured for GPU access.

**Fix:** Use the GPU docker-compose file (Path A, Option 2). The Docker Desktop GUI run flow doesn't enable GPU by default.

### Mac: `docker-compose-gpu.yml` fails or shows "No GPU"

Expected — Docker Desktop on macOS has no GPU passthrough at all (see A.2). Switch to `docker-compose-cpu.yml`. There is no driver swap or compose tweak that enables Apple Silicon or AMD GPUs inside a Docker container on macOS. For native Apple Silicon GPU acceleration, run the workload outside Docker (PyTorch MPS, MLX) on the macOS host directly.

### "CUDA Not Available" (Paths B/C)

This is normal on CPU-only systems. The app works fully on CPU. To install CUDA:

```bash
# Check current CUDA version
nvidia-smi

# If not installed, follow NVIDIA's official guide:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
```

### "401 Unauthorized" Error

**Cause:** Your device is not authorized with Ollama.

**Fix:** A connect link will appear in the chat. Click it, log in to Ollama, click "Connect". If no link appears, restart the app.

### "Model Temporarily Unavailable"

The free AI model provider is overloaded. Try again in a few minutes, or switch to a different model in Settings.

### "ModuleNotFoundError" (Paths B/C)

```bash
conda activate optimat-alloys
pip install -r requirements.txt
```

### "Reference Data Files Missing"

```bash
ls data/reference/
# Should see lattice_constants_*.json and energies_per_atom_*.json

# If empty, regenerate (takes several hours):
python scripts/run_precompute.py
```

### App Won't Start (Docker)

1. Is Docker Desktop running? (green status in bottom-left)
2. Is port 8000 free?
3. Try deleting the container and creating a new one (data persists if you used named volumes).

### Slow Performance

- **AI responses slow:** Switch to a cloud model (gpt-oss:120b-cloud)
- **Calculations slow:** Use a smaller supercell size (48 atoms)
- **Everything slow (Docker):** Docker Desktop → Settings → Resources → Memory → set to at least 8 GB
- **Everything slow (WSL2):** Verify you're on the WSL2 native filesystem (`pwd` shows `/home/...`, not `/mnt/c/...`)

---

## Updating

### Path A: Docker

```powershell
cd C:\OptiMat-Alloys
docker compose -f docker-compose-gpu.yml pull    # or -cpu.yml
docker compose -f docker-compose-gpu.yml up
```

Named volumes preserve your data across image updates.

### Paths B/C: Conda

```bash
cd ~/OptiMat-Alloys
git pull origin main
conda activate optimat-alloys
pip install -r requirements.txt
chainlit run run_chat.py
```

---

## Uninstalling

### Path A: Docker

In Docker Desktop:
1. **Containers** tab → Stop the container → Delete
2. **Images** tab → Delete `ghcr.io/optimat-chat/optimat-alloys`
3. **Volumes** tab → Delete `alloy-data` and `ollama-models` (this erases your structures)

### Paths B/C: Conda

```bash
# Remove conda environments
conda deactivate
conda env remove -n optimat-alloys
conda env remove -n optimat-nequip   # if installed

# Remove the repository
rm -rf ~/OptiMat-Alloys
```

---

## Quick Reference

| Action | How |
|--------|-----|
| Open the app | http://localhost:8000 |
| Change AI model | Gear icon → AI Model dropdown |
| Change supercell size | Gear icon → Default Supercell Size |
| Export database | Click "Download Database" button |
| Start the app (Docker) | `docker compose -f docker-compose-gpu.yml up` (or `-cpu.yml`) |
| Stop the app (Docker) | Ctrl-C, or `docker compose -f docker-compose-gpu.yml down` |
| Stop the app (Paths B/C) | Ctrl-C in terminal, or `pkill -f "chainlit run"` |

---

**Need help?** Open an issue at https://github.com/OptiMat-Chat/OptiMat-Alloys/issues
