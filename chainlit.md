# Welcome to OptiMat Alloys! 🚀🤖

**Design tomorrow's materials, today.**

OptiMat Alloys is your AI research partner — an autonomous agent that can plan, run, and analyze atomistic simulations with *near-quantum accuracy*, powered by **universal machine-learning potentials** trained on massive DFT datasets including **OMat24** (100M+ calculations).

---

## How It Works

OptiMat Alloys integrates a **Large Language Model (LLM)** with **universal neural network potentials (NNP)** in an autonomous agentic workflow:

![Agentic system concept](/public/Agentic_new.png)

The agent reasons about your request, selects the right tools, and executes simulations and analysis automatically:

![Interaction schema](/public/Interaction_schema2.png)

The potentials are trained on datasets spanning the entire periodic table, sampled far from equilibrium:

![OMat24 dataset](/public/OMat24.png)

Currently supported elements for alloy design:

![Supported elements](/public/Elements.png)

---

## Available Tools

OptiMat Alloys provides 7 computational tools that the AI agent calls automatically:

| Tool | What It Does |
|------|-------------|
| **Generate Alloy Supercell** | Creates SQS (Special Quasirandom Structure) supercells with 2-stage relaxation (coarse GPU + fine CPU) |
| **Search Database** | Finds existing structures by composition, calculator, structure type, stability, and more |
| **Calculate Elastic Properties** | Computes full elastic stiffness tensor (6×6) using finite differences, with ELATE anisotropy analysis |
| **Compute Thermal Properties (QHA)** | Quasi-Harmonic Approximation for temperature-dependent B(T), Cp(T), α(T), G(T), V(T) |
| **Generate Report** | Creates comprehensive visual report with PDF, structure files (CIF/POSCAR/XYZ), and data exports (CSV) |
| **Database Statistics** | Interactive Plotly charts showing database composition, growth, calculator distribution |
| **Recompute Structure** | Re-relaxes an existing structure with a different calculator for benchmarking |

---

## Settings & Parameters

Click the **gear icon** (⚙️) near the chat input to adjust settings.

### AI Model

| Model | Type | Description |
|-------|------|-------------|
| **gpt-oss:120b-cloud** | Ollama Cloud | Default. 120B parameter model, runs remotely. No GPU needed. |
| **gpt-oss:20b** | Ollama Local | Runs on your GPU (12GB VRAM recommended). Private, no internet needed for AI. |
| **GLM-4.5-Air** | OpenRouter Free | 106B MoE model via OpenRouter. Free, may hit rate limits. |
| **GPT-OSS 120B/20B** | OpenRouter Free | OpenAI open-source models via OpenRouter. Free. |
| **Qwen3-Coder** | OpenRouter Free | 480B MoE coder model via OpenRouter. Free, 50 req/day limit. |

**OpenRouter rate limits:** Free models may hit provider rate limits (50 requests/day without credits). To unlock unlimited rate limits:
1. Deposit $10+ on your OpenRouter account — this will **not** be spent on free models, it just removes the rate limit cap
2. Optionally, provide your own API key via the BYOK (Bring Your Own Key) settings at https://openrouter.ai/settings/integrations for dedicated rate limits

### Force Field Calculator

| Calculator | Accuracy | Speed | Best For |
|-----------|----------|-------|----------|
| **ORB v3 Conservative** | High | Fast | Default — good balance for most alloys |
| **ORB v3 Direct** | High | Fast | Alternative ORB variant |
| **MACE-OMAT Medium** | Very High | Medium | High-accuracy studies, phonons |
| **MACE-MPA Medium** | Very High | Medium | Materials Project-trained variant |
| **NequIP OAM-L** | High | Slow | Equivariant neural network potential |
| **NequIP OAM-XL** | Highest | Slowest | Best accuracy, most compute-intensive |

All calculators use the same workflow (SQS → relaxation → analysis) but differ in accuracy and speed. Results from different calculators can be compared using the Recompute tool.

**Note:** NequIP calculators cannot handle structures larger than ~500 atoms due to memory constraints (tested on NVIDIA RTX 5000 Ada, 16GB VRAM, 64GB RAM). May work on more powerful workstations with higher RAM. Use ORB or MACE for large supercells.

### Default Supercell Size

| Size | Atoms | Use Case |
|------|-------|----------|
| **Small (~48 atoms)** | ~48 | Quick exploration, QHA calculations, rapid screening |
| **Medium (~500 atoms)** | ~500 | Balanced — good statistics and reasonable compute time |
| **Large (~2048 atoms)** | ~2048 | Best statistical accuracy, slowest |

**Note:** These are target atom counts. The actual system size may vary slightly depending on the crystal structure and composition (e.g., a "48-atom" FCC cell may have 48 or 54 atoms).

**Computational constraints by supercell size:**

| Calculation | ~48 atoms | ~500 atoms | ~2048 atoms |
|-------------|----------|-----------|------------|
| Structure generation | Fast | Moderate | Slow |
| Elastic properties | Fast | Moderate | **Extremely slow** (6×6 tensor, many force evaluations) |
| QHA thermal properties | Recommended | **Not recommended** (extremely slow + memory issues) | **Not feasible** |

These times depend on your hardware. As a reference:
- **NVIDIA RTX 5000 Ada (16GB VRAM, 64GB RAM):** Handles 48-atom calculations comfortably for all property types. Handles ~500-atom structure generation and elastic calculations, but QHA on ~500 atoms will trigger memory errors.
- **NVIDIA GH200 (4 GPUs, ~96GB VRAM):** QHA on ~500 atoms took approximately **16 hours** to complete.

QHA calculations are designed for small supercells. For ~500+ atoms, consider computing only elastic and structural properties. Evaluate your device's capabilities when choosing supercell size and calculator.

---

## What You Can Do

### Structure Design
- Generate random solid solution alloys (binary to senary+)
- Specify composition, crystal structure (FCC, BCC, HCP), and atom count
- Automatic SQS optimization for representative disorder

### Property Calculation
- **Formation energy** — thermodynamic stability relative to pure elements
- **Elastic properties** — bulk/shear/Young's modulus, Poisson's ratio, anisotropy
- **ELATE analysis** — directional Young's modulus, shear modulus, Poisson's ratio (2D projections + 3D surfaces)
- **Thermal properties (QHA)** — temperature-dependent bulk modulus, heat capacity, thermal expansion, Gibbs free energy (0–600 K)
- **Structural analysis** — PTM classification, RDF, density, coordination

### Data Management
- Search and filter database by composition, calculator, stability
- Export structures (CIF, POSCAR, LAMMPS, XYZ)
- Generate PDF reports with all visualizations
- Download entire database for backup or sharing

---

## Example Queries

Try typing these in the chat:

- `Generate a FCC Cu50Ni50 alloy with 48 atoms`
- `What are the elastic properties of CoCrFeNi?`
- `Calculate thermal properties for structure 141`
- `Search for BCC alloys with Co and Cr`
- `Show me a report for structure 242`
- `Show database statistics`
- `Compare the elastic properties of FCC and BCC CoCrFeNi`

---

**Happy simulating! 💻✨**
