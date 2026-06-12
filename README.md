# OptiMat Alloys

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/OptiMat-Chat/OptiMat-Alloys/releases/tag/v1.0.0)
[![Docker image](https://github.com/OptiMat-Chat/OptiMat-Alloys/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/OptiMat-Chat/OptiMat-Alloys/actions/workflows/docker-publish.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2604.21850-b31b1b.svg)](https://arxiv.org/abs/2604.21850)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-early%20access-brightgreen.svg)]()

An AI-powered materials science research tool that combines LLMs with universal neural network potentials (ORB, MACE, NequIP) to perform atomistic simulations with near-DFT accuracy.

**Focus**: Multi-principal element alloy design with structural stability analysis, elastic property prediction, and comprehensive materials characterization.

**Preprint**: [OptiMat Alloys: a FAIR, living database of multi-principal element alloys enabled by a conversational agent](https://arxiv.org/abs/2604.21850) (Hu & Turlo, arXiv:2604.21850).

## 🚀 Quick Start — Docker (recommended)

The fastest way to run OptiMat Alloys on **Windows, macOS, or Linux** is the official Docker image, built automatically from this repository:

```
ghcr.io/optimat-chat/optimat-alloys
```

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux). No clone, no conda, no Python setup required.

### 1. Get a compose file

Download the compose file for your hardware into any folder:

```bash
# CPU only (works everywhere)
curl -LO https://raw.githubusercontent.com/OptiMat-Chat/OptiMat-Alloys/main/docker-compose-cpu.yml

# NVIDIA GPU (5-10x faster simulations)
curl -LO https://raw.githubusercontent.com/OptiMat-Chat/OptiMat-Alloys/main/docker-compose-gpu.yml
```

(Or simply copy the file contents from this repo via your browser.)

### 2. Start the container

```bash
docker compose -f docker-compose-cpu.yml up    # CPU
docker compose -f docker-compose-gpu.yml up    # NVIDIA GPU
```

### 3. Open the app

Go to **http://localhost:8000** in your browser. Enter your API key in the chat UI at startup (it is saved for next time), or create a `.env` file next to the compose file before starting (works the same on Windows, macOS, and Linux):

```bash
# .env — in the same folder as the compose file
OLLAMA_API_KEY=your-key-here
```

Supported providers: `OLLAMA_API_KEY` (cloud models, default), `OPENROUTER_API_KEY` (free models available), `OPENAI_API_KEY`. See [Configuration](#%EF%B8%8F-configuration).

> **Pinning a version:** the compose files track `:latest`. For a reproducible install, edit the `image:` line to a release tag, e.g. `ghcr.io/optimat-chat/optimat-alloys:1.0.0`.

### Alternative: plain `docker run`

```bash
docker run --gpus all -p 8000:8000 \
  -v alloy-data:/app/structures -v ollama-models:/root/.ollama \
  ghcr.io/optimat-chat/optimat-alloys:latest
```

(Drop `--gpus all` on machines without an NVIDIA GPU.)

### Good to know

- **First run**: the system precomputes reference data for 117 elements on first launch — this takes **several hours** (faster with GPU) and happens **once per calculator**; subsequent launches are instant.
- **Your data persists** in named Docker volumes (`alloy-data` for structures, `ollama-models` for local LLM weights) and survives container restarts and updates.
- **Update**: `docker compose -f docker-compose-cpu.yml pull` then `up`.
- **Remove**: `docker compose -f docker-compose-cpu.yml down` (add `-v` to also delete your data volumes).
- **Step-by-step Windows walkthrough** (Docker Desktop, WSL2 backend, GPU passthrough): [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

## ☁️ Cloud Version

A read-only cloud deployment for exploring 1000+ pre-computed community structures — no installation, no GPU, any device.

**Access:** Coming soon at `https://app.optimat.chat`

## 🔧 Install from Source (for developers)

For developers who want full control or need to modify the code. Requires **Linux or WSL2** (Windows Subsystem for Linux 2) — see the [WSL2 setup guide](docs/SETUP_GUIDE.md#path-b-wsl2--conda-windows-developers).

```bash
# 1. Clone the repository (use the WSL2 native filesystem for best performance)
cd ~
git clone https://github.com/OptiMat-Chat/OptiMat-Alloys.git
cd OptiMat-Alloys

# 2. Run the setup script
bash scripts/setup_linux.sh

# The script will:
# - Create the optimat-alloys conda environment with Python 3.11
# - Install all dependencies
# - Optionally configure CUDA support
# - Optionally install cuML for large systems (5k+ atoms)
# (NequIP calculators need a second env — see docs/SETUP_GUIDE.md#nequip-support-optional)

# 3. Set your API key
cp .env.example .env
nano .env   # add OLLAMA_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY

# 4. Launch OptiMat Alloys
bash scripts/launch_optimat_alloys.sh
```

The application will open in your browser at `http://localhost:8000`.

Details, hardware notes, and troubleshooting: [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

## 🖥️ Platform Support

| Install method | Windows | macOS | Linux |
|---|---|---|---|
| **Docker** (recommended) | ✅ Docker Desktop | ✅ Docker Desktop | ✅ Docker Engine |
| **From source** | ✅ via WSL2 | ❌ | ✅ native |

**System requirements:** 16 GB+ RAM recommended; NVIDIA GPU with 6 GB+ VRAM optional but recommended (5-10x faster simulations); one API key (Ollama cloud, OpenRouter free tier, or OpenAI).

## ✨ Features

### Core Capabilities
- **Agentic AI System**: Autonomous task execution with 7 specialized tools
- **Interactive Web Interface**: Powered by Chainlit for real-time collaboration
- **High-Accuracy Simulations**: ORB universal potentials trained on 100M+ DFT calculations
- **Multi-Principal Alloys**: Design and analyze complex alloy systems
- **Structural Analysis**: PTM (Polyhedral Template Matching) for phase identification
- **Elastic Property Prediction**: Full elastic tensor calculation with ELATE anisotropy analysis
- **Visualization**: OVITO-rendered images and interactive Plotly charts
- **Global Structure Database**: SQLite database with searchable composition metadata

### v1.0.0 Enhancements
- **ELATE Integration**: Comprehensive elastic anisotropy analysis with directional property visualizations
- **QHA Temperature Properties**: Compute B(T), V(T), α(T), Cp(T), and optionally κ(T)
- **Cooperative Cancellation**: Stop long-running tasks (elastic calculations) with progress feedback
- **Configurable Settings**: Calculator selection (8 calculators), supercell sizes (96/512/2048 atoms), logging verbosity
- **Total RDF Plots**: Correctly weighted radial distribution functions
- **Precise Database Search**: `exact_elements_only` filter for composition matching
- **Enhanced Agent Intelligence**: Tools return rich metadata for data-driven recommendations

## 📚 Documentation

The install instructions above are the canonical ones. For deep-dives:

- **[Setup Guide](docs/SETUP_GUIDE.md)** — appendix: WSL2 setup, GPU enablement, hardware notes, verification, troubleshooting
- **[Configuration](docs/CONFIGURATION.md)** — models, calculators, settings
- **[Element Support](docs/ELEMENT_SUPPORT.md)** — per-calculator element coverage
- **[Maintenance](docs/MAINTENANCE.md)** — database and cache management

## 🏗️ Architecture

OptiMat Alloys uses a modular architecture:

```
src/
├── core/              # Pure business logic
│   ├── calculators.py        # ORB model management
│   ├── structure_builder.py  # Alloy supercell generation
│   ├── optimization.py       # Structure relaxation
│   ├── analysis.py           # PTM, RDF, density
│   ├── elasticity.py         # Elastic tensor calculation
│   ├── elate_analysis.py     # Anisotropy analysis (ELATE)
│   └── formation_energy.py   # Formation energy
├── visualization/     # OVITO and Plotly rendering
│   ├── ovito_renderer.py     # Structure visualization
│   ├── plotly_charts.py      # RDF and analysis charts
│   ├── elate_plots.py        # Elastic property visualizations
│   └── database_charts.py    # Database analytics dashboard
├── storage/          # Data persistence
│   ├── global_database.py    # SQLite structure database
│   └── cache.py              # Reference data caching
├── agents/           # AutoGen agent system
│   ├── factory.py            # Agent creation
│   └── scientist.py          # Scientist agent
└── tools/            # Agent tool infrastructure
```

**Main file**: `run_chat.py` (~691 lines)

**7 Agent Tools**:
1. `generate_alloy_supercell` - Build and relax alloy structures
2. `search_database` - Search structures by composition
3. `calculate_elastic_properties` - Compute elastic tensors and anisotropy
4. `compute_anharmonic_properties` - Temperature-dependent properties (QHA)
5. `visualize_database_statistics` - Database analytics dashboard
6. `generate_report` - Comprehensive reports with PDF/CSV export
7. `recompute_structure` - Re-relax with different calculator for benchmarking

## 🔧 Usage Examples

### Generate an Alloy Supercell

```
User: "Create a CuAuAg equiatomic alloy with 512 atoms in an FCC structure"

OptiMat Alloys will:
1. Calculate optimal supercell dimensions
2. Generate SQS (Special Quasirandom Structure)
3. Relax the structure using ORB potential
4. Perform PTM structural analysis
5. Calculate formation energy and density
6. Render visualization and save to database
```

### Analyze Structural Stability

```
User: "Compare the stability of TiZrHf in BCC vs FCC structures"

OptiMat Alloys will:
1. Build both structures
2. Relax to equilibrium
3. Calculate formation energies
4. Analyze local structure with PTM
5. Provide stability comparison
```

### Calculate Elastic Properties

```
User: "Calculate the elastic tensor and analyze anisotropy for CuAg"

OptiMat Alloys will:
1. Apply finite difference deformations (180 calculations)
2. Compute 6×6 elastic tensor (Voigt notation)
3. Calculate Voigt/Reuss/Hill averages for K, G, E, ν
4. Analyze anisotropy (Universal Anisotropy Index)
5. Predict ductility (Pugh ratio K/G)
6. Generate 2D/3D directional property visualizations
7. Calculate acoustic wave speeds
```

### Compute Temperature-Dependent Properties

```
User: "Calculate thermal properties for CuAg alloy"

OptiMat Alloys will:
1. Generate strained volumes (±10% around equilibrium)
2. Compute phonon frequencies at each volume
3. Run QHA analysis via Phonopy
4. Calculate B(T), V(T), α(T), Cp(T), γ(T)
5. Generate interactive temperature plots
6. Optionally compute thermal conductivity κ(T)

Currently supports calculations from 0-600K.
```

## 🧪 Supported Elements

Element support varies by calculator family:

| Calculator Family | Elements | Coverage |
|-------------------|----------|----------|
| **ORB** (Orbital Materials) | 117/118 | 99.2% - All elements except Og |
| **MACE** (Foundation) | 89/118 | 75.4% - H-Pu, excludes late actinides |
| **NequIP** (Foundation) | 86 | H through Pu (Z=1-94) |

**ORB Full Support**: All transition metals, lanthanides, actinides, nonmetals (H, C, N, O, etc.), noble gases, superheavy elements (Rf-Ts).

**MACE/NequIP**: Most main group elements, all transition metals, lanthanides (La-Lu). See docs for full list.

See [`docs/ELEMENT_SUPPORT.md`](docs/ELEMENT_SUPPORT.md) for detailed testing results.

## 🔬 Technical Stack

- **AI Framework**: AutoGen (multi-agent orchestration)
- **Web Interface**: Chainlit (interactive chat UI)
- **Atomistic Simulation**: ASE (Atomic Simulation Environment)
- **Neural Network Potentials**: ORB, MACE, NequIP (universal ML potentials)
- **Finite-Temperature**: Phonopy QHA (B(T), V(T), α(T), Cp(T)), optional phono3py for κ(T)
- **Structural Analysis**: OVITO with PTM
- **Elastic Analysis**: MechElastic ELATE (anisotropy analysis)
- **Visualization**: Plotly charts, OVITO Tachyon/Anari rendering
- **Alloy Generation**: sqsgenerator (SQS structures)
- **Database**: SQLite with searchable metadata

## ⚙️ Configuration

### API Keys

OptiMat Alloys supports multiple API providers:

**Ollama Cloud (default)**
```bash
export OLLAMA_API_KEY='...'
# Sign up at https://ollama.com/ — cloud models need no local GPU
```

**OpenRouter (FREE cloud models)** - Recommended for getting started
```bash
export OPENROUTER_API_KEY='sk-or-...'
# Free signup at https://openrouter.ai/keys
```

**OpenAI (paid cloud models)**
```bash
export OPENAI_API_KEY='sk-...'
```

**Ollama (local models)** - No API key required
```bash
# Just install Ollama and run: ollama serve
# Models auto-detected in settings dropdown
```

You can also enter API keys via the Chainlit UI at startup (automatically saved to .env).

### Model Configuration

**Cloud Models (OpenAI)**:
- gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
- Requires API key

**Local Models (Ollama)**:
- gpt-oss:20b (preferred), qwen2.5:14b, mistral-small:24b
- Requires Ollama installed locally (bundled in the Docker image)

Select via unified dropdown in Chainlit settings. Models can be switched mid-session.

### Calculator Selection

Select ML potential calculator via Chainlit settings UI:

**ORB (Orbital Materials)** - Default, trained on OMat24:
- `orb-v3-conservative-inf-omat` - Most accurate, NVE MD compatible
- `orb-v3-direct-20-omat` (default) - 2-3x faster

**NequIP (Foundation models)**:
- `nequip-oam-xl` - Highest accuracy (F1: 0.906)
- `nequip-oam-l` - Balanced, most tested
- `nequip-mp-l` - MPtrj compatibility

**MACE (Foundation models)**:
- `mace-mpa-0-medium` - MPTrj + sAlex trained (Matbench SOTA)
- `mace-omat-0-small/medium` - OMAT trained (best for phonons)

Each structure remembers which calculator was used for reproducibility.
Use `recompute_structure` tool to benchmark calculators on the same atomic configuration.

### Supercell Size Configuration

Configure target supercell size in Chainlit settings:
- **Small**: 48 atoms (default, fast prototyping)
- **Medium**: 512 atoms (balanced)
- **Large**: 2048 atoms (detailed analysis)

### Logging Configuration

Control console verbosity with `.env` file:
```bash
# Clean console output (default, recommended)
LOG_LEVEL=WARNING

# Full debugging output
LOG_LEVEL=DEBUG
```

## 🎯 Performance Tips

### GPU Acceleration (NVIDIA)

```bash
# Verify CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Simulations are **5-10x faster** with GPU.

### Large Systems (5k+ atoms)

For systems with ≥5000 atoms (PBC) or ≥30000 atoms (non-PBC), install cuML (source installs):

```bash
conda activate optimat-alloys

# CUDA 11.4-11.8
pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu11==25.2.*"

# CUDA 12.0+
pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.2.*"
```

Benefits: 2-10x faster graph creation, 2-100x better GPU memory efficiency

### File Performance (WSL2)

**Always use the WSL2 native filesystem** (source installs):

```bash
# FAST ✅
~/OptiMat-Alloys/

# SLOW ❌ (10-100x slower)
/mnt/c/Users/YourName/Desktop/OptiMat-Alloys/
```

## 📁 Output Files

### Global Structure Database

All structures are stored in a centralized SQLite database:

```
structures/
├── database.db                         # SQLite database (all structures)
├── {uuid}/                             # Per-structure directory (UUID-based)
│   ├── structure_elements.png         # Element-colored rendering
│   ├── structure_analysis.png         # PTM-colored rendering
│   ├── relaxation.traj                # Optimization trajectory
│   ├── rdf_chart.json                 # RDF Plotly chart data
│   └── elastic_tensor.json            # Elastic properties (if calculated)
```

**Example**: `structures/f1f24f2b5e584926ab47cb49ac2591d6/`

In Docker, this directory lives in the `alloy-data` volume mounted at `/app/structures`.

### Reference Data

Calculator-specific reference data (computed once per calculator):

```
data/reference/
├── lattice_constants_orb_v3_direct_20_omat.json
├── energies_per_atom_orb_v3_direct_20_omat.json
├── lattice_constants_mace_mpa_0_medium.json
├── energies_per_atom_mace_mpa_0_medium.json
├── lattice_constants_nequip_oam_l.json
├── energies_per_atom_nequip_oam_l.json
└── ... (files for all 8 calculators)
```

### Database Features
- **UUID-based storage**: ASE's built-in UUIDs (32-character hex strings) for global uniqueness
- **Cloud compatibility**: PostgreSQL-compatible UUIDs for future cloud deployment
- **Searchable metadata**: 118 element composition fractions (all periodic table), calculator info
- **Immutable structures**: Calculator and settings preserved for reproducibility

### Report Export

The `generate_report` tool produces comprehensive, publication-ready outputs:

- **PDF Report**: Multi-page document with structure visualizations, property tables, and all analysis figures
- **Data ZIP**: Contains CIF/POSCAR/XYZ structure files, CSV data tables, and BibTeX references
- **Interactive Charts**: Plotly visualizations displayed in chat and embedded in PDF

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Docker: map a different host port
docker run -p 8001:8000 ghcr.io/optimat-chat/optimat-alloys:latest

# Source install:
bash scripts/launch_optimat_alloys.sh --port 8001
```

### CUDA Out of Memory
- Close other GPU applications
- Reduce `target_num_atoms` (try 256 instead of 512)
- Install cuML for better memory efficiency

### Slow Performance
- Verify you're on WSL2 native filesystem (`pwd` shows `/home/...`)
- Check GPU usage: `nvidia-smi`
- Ensure conda environment is activated (source installs)

### Import Errors (source installs)
```bash
conda activate optimat-alloys
pip install --upgrade -r requirements.txt
```

More: [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

## 🔄 Updating

**Docker:**
```bash
docker compose -f docker-compose-cpu.yml pull
docker compose -f docker-compose-cpu.yml up
```

**Source install:**
```bash
cd ~/OptiMat-Alloys
git pull
conda activate optimat-alloys
pip install --upgrade -r requirements.txt
```

## 🧹 Uninstalling

**Docker:**
```bash
docker compose -f docker-compose-cpu.yml down        # keep your data
docker compose -f docker-compose-cpu.yml down -v     # delete data volumes too
docker rmi ghcr.io/optimat-chat/optimat-alloys:latest
```

**Source install:**
```bash
conda deactivate
conda env remove -n optimat-alloys
rm -rf ~/OptiMat-Alloys
```

## 🤝 Contributing

Contributions welcome! This is a research tool in early access.

## 📄 Citation

If you use OptiMat Alloys in your research, please cite the preprint:

```bibtex
@misc{hu2026optimatalloys,
  title         = {OptiMat Alloys: a FAIR, living database of multi-principal
                   element alloys enabled by a conversational agent},
  author        = {Hu, Yang and Turlo, Vladyslav},
  year          = {2026},
  eprint        = {2604.21850},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  url           = {https://arxiv.org/abs/2604.21850}
}
```

## 📜 License

[MIT](LICENSE)

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/OptiMat-Chat/OptiMat-Alloys/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OptiMat-Chat/OptiMat-Alloys/discussions)
- **Website**: [optimat.chat](https://optimat.chat/agents/optimat-alloys)

## 🙏 Acknowledgments

- **ORB Models**: Orbital Materials (OMat24 dataset)
- **MACE**: Ilyes Batatia et al. (Cambridge)
- **NequIP**: Simon Batzner et al. (Harvard)
- **Phonopy**: Atsushi Togo (QHA implementation)
- **ASE**: Atomic Simulation Environment team
- **OVITO**: Alexander Stukowski
- **AutoGen**: Microsoft Research
- **Chainlit**: Chainlit team

---

**Built for materials scientists, by materials scientists.** 🔬⚡
