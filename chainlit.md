# Welcome to OptiMat Alloys! 🚀🤖

**Design tomorrow's materials, today.**

OptiMat Alloys is your AI research partner — an autonomous agent that can plan, run, and analyze atomistic simulations with *near-quantum accuracy*, powered by **universal machine-learning potentials** trained on massive DFT datasets including **OMat24** (100M+ calculations).

---

## How It Works

OptiMat Alloys integrates a **Large Language Model (LLM)** with **universal neural network potentials (NNP)** in an autonomous agentic workflow:

![Agentic system concept](/public/Agentic_new.png)

The agent reasons about your request, selects the right tools, and executes simulations and analysis automatically:

![Interaction schema](/public/Interaction_schema2.png)

The potentials are trained on datasets that reach across the periodic table, sampled far from equilibrium — though the coverage is very uneven:

![OMat24 dataset](/public/OMat24.png)

How well each element is represented in that training data:

![Elemental representation in the OMat24 dataset](/public/Elements.png)

**Colour** is the element's fraction of the dataset (log scale) — pale cells are
barely present, black cells essentially absent. **Blue outlines** mark the metals
this app is built for; those are the elements to reach for.

Coverage is not the same as accuracy. A calculator will happily run on an element
it hardly saw in training, and return a number that looks ordinary — the reference
energies for the elements past Pu come back positive, which is not a cohesive
energy at all. In practice, treat **Z ≤ 96** as the limit on
`orb-v3-conservative-inf-omat`, and **Z ≤ 94** on MACE and NequIP. Build and
visualise anything you like; be careful quoting energies outside those ranges.

Light non-metals (H, He, N, O, F, Ne, Cl, Ar) have reference energies computed
with atoms held on their lattice sites, since they would otherwise relax into
molecules. Those are valid for **mixing energy**; a formation energy against them
is not comparable with the usual convention, which references H₂ rather than a
hydrogen crystal.

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
| **gpt-oss:120b-cloud** | Ollama Cloud | Default. 120B parameters, runs remotely. No GPU needed. |
| **gpt-oss:20b** | Ollama Local | Runs on your GPU (12GB VRAM recommended). Private, no internet needed for AI. |
| *(varies)* | OpenRouter Free | Discovered at startup — see below |

The OpenRouter entries are **not a fixed list**. Providers retire free model IDs
without notice, so the app queries OpenRouter each session for models that are
free and support tool calling, then orders them by measured ability on this
app's own tools: a suite that drives all seven tools through the real agent and
scores whether each was called correctly. Best-measured models appear first, and
the dropdown is capped at eight.

The Ollama Cloud default is first for a reason — it is the one option with no
provider rate limit and no retirement risk. Prefer it for long sessions.

**OpenRouter rate limits.** Free models are capped at **20 requests per minute**
and, per UTC day, **50 requests** if you have purchased under 10 credits or
**1000** at 10 or more. Depositing $10 raises the daily cap; it is not spent on
free models, and it does not lift the per-minute limit. You can also supply your
own provider keys via BYOK at https://openrouter.ai/settings/integrations.

If a model becomes unavailable mid-session the app says so and suggests
switching, rather than failing silently.

### Force Field Calculator

| Calculator | Accuracy | Speed | Best For |
|-----------|----------|-------|----------|
| **ORB v3 Conservative** | High | Fast | Default — good balance for most alloys |
| **ORB v3 Direct** | High | Fast | Structures and energies only — **not elastic constants** (see below) |
| **NequIP OAM-L** | High | Slow | Equivariant neural network potential |
| **NequIP OAM-XL** | Highest | Slowest | Best accuracy, most compute-intensive |
| **NequIP MP-L** | Moderate | Slow | Materials Project training data only |
| **MACE-MPA Medium** | Very High | Medium | Materials Project-trained variant |
| **MACE-OMAT Medium** | Very High | Medium | High-accuracy studies, phonons |
| **MACE-OMAT Small** | High | Fast | Faster MACE-OMAT variant |

All calculators use the same workflow (SQS → relaxation → analysis) but differ in accuracy and speed. Results from different calculators can be compared using the Recompute tool.

**Independent benchmarks.** [Matbench Discovery](https://matbench-discovery.materialsproject.org)
ranks universal potentials on materials discovery, geometry optimisation and phonons:

| Calculator | Benchmark page |
|---|---|
| ORB v3 Conservative / Direct | *not linked — see note* |
| NequIP OAM-L | [nequip-oam-l-0.1](https://matbench-discovery.materialsproject.org/models/nequip-oam-l-0.1) |
| NequIP OAM-XL | [nequip-oam-xl-0.1](https://matbench-discovery.materialsproject.org/models/nequip-oam-xl-0.1) |
| NequIP MP-L | [nequip-mp-l-0.1](https://matbench-discovery.materialsproject.org/models/nequip-mp-l-0.1) |
| MACE-MPA Medium | [mace-mpa-0](https://matbench-discovery.materialsproject.org/models/mace-mpa-0) |
| MACE-OMAT Medium / Small | *not benchmarked there* |

Two calculators are deliberately left unlinked, because no page there measures
what we ship:

- **ORB v3.** The leaderboard entry benchmarks
  `orb-v3-conservative-inf-mpa-20250404.ckpt`; this app ships the `-omat-`
  checkpoint — same date and formulation, different training set. The **Direct**
  variant is not benchmarked at all. Citing those numbers for our ORB results
  would attribute someone else's checkpoint to them.
- **MACE-OMAT** has no entry there in either size.

The NequIP entries and MACE-MPA-0 do correspond to the checkpoints used here.

**⚠️ ORB v3 Direct and elastic constants.** The elastic tensor is obtained from
the energy response to strain, which assumes forces are the exact energy gradient.
ORB's "direct" models predict forces from a separate head, so their energy surface
is not quadratic in strain and the fitted constants come out several times too
large — measured on a 32-atom fcc Cu50Ni50: C₁₁ = 1057 GPa against 228 GPa from
the conservative model, with literature near 200. The app warns you before
computing elastic properties with such a model. Use **ORB v3 Conservative**,
MACE or NequIP for elasticity.

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
