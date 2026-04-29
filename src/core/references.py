"""Computational methods documentation and references.

This module provides a centralized repository of:
1. Computational method descriptions
2. BibTeX references for citations
3. Software version information

Used by the report generator to create publication-ready documentation.
"""

from typing import Dict, List, Any, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComputationalMethod:
    """Description of a computational method."""
    name: str
    description: str
    parameters: Dict[str, str]
    references: List[str]


class ReferenceTracker:
    """Track citations in order of first appearance for numbered citation format.

    This class provides a way to manage citations throughout a document,
    assigning numbers based on the order references are first cited.
    """

    def __init__(self):
        """Initialize an empty reference tracker."""
        self._cited_keys: List[str] = []

    def cite(self, ref_keys: Union[str, List[str]]) -> str:
        """Register citation(s) and return numbered format.

        Args:
            ref_keys: Single reference key or list of keys
                     (e.g., "ptm_larsen2016" or ["voigt1928", "reuss1929"])

        Returns:
            Formatted string like "[1]" or "[1, 2, 3]"
        """
        if isinstance(ref_keys, str):
            ref_keys = [ref_keys]

        numbers = []
        for key in ref_keys:
            if key not in self._cited_keys:
                self._cited_keys.append(key)
            numbers.append(self._cited_keys.index(key) + 1)

        numbers.sort()
        return "[" + ", ".join(str(n) for n in numbers) + "]"

    def get_ordered_keys(self) -> List[str]:
        """Get reference keys in citation order.

        Returns:
            List of reference keys in the order they were first cited
        """
        return self._cited_keys.copy()

    def get_reference_list(self) -> List[str]:
        """Get formatted reference list in citation order.

        Returns:
            List of formatted reference strings with numbers
        """
        return get_publication_references(self._cited_keys)


# BibTeX references for all methods and software
BIBTEX_REFERENCES = {
    "orb2024": """@misc{orb2024,
    author = {{Orbital Materials}},
    title = {{ORB Models: Universal Neural Network Potentials for Atomistic Simulations}},
    year = {2024},
    url = {https://github.com/orbital-materials/orb-models},
    note = {GitHub repository}
}""",

    "omat24_2024": """@article{omat24_2024,
    author = {Barroso-Luque, Luis and others},
    title = {{Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models}},
    journal = {arXiv preprint},
    year = {2024},
    eprint = {2410.12771},
    archiveprefix = {arXiv},
    primaryclass = {cond-mat.mtrl-sci}
}""",

    "phonopy2015": """@article{phonopy2015,
    author = {Togo, Atsushi and Tanaka, Isao},
    title = {{First principles phonon calculations in materials science}},
    journal = {Scripta Materialia},
    volume = {108},
    pages = {1--5},
    year = {2015},
    doi = {10.1016/j.scriptamat.2015.07.021}
}""",

    "qha_togo2010": """@article{qha_togo2010,
    author = {Togo, Atsushi and Chaput, Laurent and Tanaka, Isao and Hug, Gilles},
    title = {{First-principles phonon calculations of thermal expansion in Ti$_3$SiC$_2$, Ti$_3$AlC$_2$, and Ti$_3$GeC$_2$}},
    journal = {Physical Review B},
    volume = {81},
    pages = {174301},
    year = {2010},
    doi = {10.1103/PhysRevB.81.174301}
}""",

    "phono3py2015": """@article{phono3py2015,
    author = {Togo, Atsushi and Chaput, Laurent and Tanaka, Isao},
    title = {{Distributions of phonon lifetimes in Brillouin zones}},
    journal = {Physical Review B},
    volume = {91},
    pages = {094306},
    year = {2015},
    doi = {10.1103/PhysRevB.91.094306}
}""",

    "elastic_fd_2002": """@article{elastic_fd_2002,
    author = {Le Page, Y. and Saxe, P.},
    title = {{Symmetry-general least-squares extraction of elastic data for strained materials from ab initio calculations of stress}},
    journal = {Physical Review B},
    volume = {65},
    pages = {104104},
    year = {2002},
    doi = {10.1103/PhysRevB.65.104104}
}""",

    "ptm_larsen2016": """@article{ptm_larsen2016,
    author = {Larsen, Peter Mahler and Schmidt, Soren and Schiotz, Jakob},
    title = {{Robust structural identification via polyhedral template matching}},
    journal = {Modelling and Simulation in Materials Science and Engineering},
    volume = {24},
    pages = {055007},
    year = {2016},
    doi = {10.1088/0965-0393/24/5/055007}
}""",

    "elate_gaillac2016": """@article{elate_gaillac2016,
    author = {Gaillac, Romain and Pullumbi, Pluton and Coudert, Fran\\c{c}ois-Xavier},
    title = {{ELATE: an open-source online application for analysis and visualization of elastic tensors}},
    journal = {Journal of Physics: Condensed Matter},
    volume = {28},
    pages = {275201},
    year = {2016},
    doi = {10.1088/0953-8984/28/27/275201}
}""",

    "ase2017": """@article{ase2017,
    author = {Larsen, Ask Hjorth and others},
    title = {{The atomic simulation environment—a Python library for working with atoms}},
    journal = {Journal of Physics: Condensed Matter},
    volume = {29},
    pages = {273002},
    year = {2017},
    doi = {10.1088/1361-648X/aa680e}
}""",

    "ovito2010": """@article{ovito2010,
    author = {Stukowski, Alexander},
    title = {{Visualization and analysis of atomistic simulation data with OVITO–the Open Visualization Tool}},
    journal = {Modelling and Simulation in Materials Science and Engineering},
    volume = {18},
    pages = {015012},
    year = {2010},
    doi = {10.1088/0965-0393/18/1/015012}
}""",

    "sqs_zunger1990": """@article{sqs_zunger1990,
    author = {Zunger, Alex and Wei, S.-H. and Ferreira, L. G. and Bernard, James E.},
    title = {{Special quasirandom structures}},
    journal = {Physical Review Letters},
    volume = {65},
    pages = {353},
    year = {1990},
    doi = {10.1103/PhysRevLett.65.353}
}""",

    "sqsgenerator2023": """@article{sqsgenerator2023,
    author = {Gehringer, Dominik and Fri{\\'{a}}k, Martin and Holec, David},
    title = {{Models of configurationally-complex alloys made simple}},
    journal = {Computer Physics Communications},
    volume = {286},
    pages = {108664},
    year = {2023},
    doi = {10.1016/j.cpc.2023.108664}
}""",

    "fire_optimizer2006": """@article{fire_optimizer2006,
    author = {Bitzek, Erik and Koskinen, Pekka and G\\"ahler, Franz and Moseler, Michael and Gumbsch, Peter},
    title = {{Structural relaxation made simple}},
    journal = {Physical Review Letters},
    volume = {97},
    pages = {170201},
    year = {2006},
    doi = {10.1103/PhysRevLett.97.170201}
}""",

    "mechelastic2021": """@article{mechelastic2021,
    author = {Singh, Sobhit and Valencia-Jaime, Irais and Pavlic, Olivia and Romero, Aldo H.},
    title = {{MechElastic: A Python library for analysis of mechanical and elastic properties of bulk and 2D materials}},
    journal = {Computer Physics Communications},
    volume = {267},
    pages = {108068},
    year = {2021},
    doi = {10.1016/j.cpc.2021.108068}
}""",

    "voigt1928": """@book{voigt1928,
    author = {Voigt, Woldemar},
    title = {{Lehrbuch der Kristallphysik}},
    publisher = {Teubner},
    address = {Leipzig},
    year = {1928}
}""",

    "reuss1929": """@article{reuss1929,
    author = {Reuss, A.},
    title = {{Berechnung der Flie{\\ss}grenze von Mischkristallen auf Grund der Plastizit{\\\"a}tsbedingung f{\\\"u}r Einkristalle}},
    journal = {Zeitschrift f{\\\"u}r Angewandte Mathematik und Mechanik},
    volume = {9},
    pages = {49--58},
    year = {1929},
    doi = {10.1002/zamm.19290090104}
}""",

    "hill1952": """@article{hill1952,
    author = {Hill, R.},
    title = {{The Elastic Behaviour of a Crystalline Aggregate}},
    journal = {Proceedings of the Physical Society. Section A},
    volume = {65},
    pages = {349--354},
    year = {1952},
    doi = {10.1088/0370-1298/65/5/307}
}""",

    "anisotropy_ranganathan2008": """@article{anisotropy_ranganathan2008,
    author = {Ranganathan, Shivakumar I. and Ostoja-Starzewski, Martin},
    title = {{Universal Elastic Anisotropy Index}},
    journal = {Physical Review Letters},
    volume = {101},
    pages = {055504},
    year = {2008},
    doi = {10.1103/PhysRevLett.101.055504}
}""",

    # MACE references
    "mace2022": """@inproceedings{mace2022,
    author = {Batatia, Ilyes and Kov{\\'a}cs, D{\\'a}vid P{\\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\\'a}nyi, G{\\'a}bor},
    title = {{MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields}},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {35},
    pages = {11423--11436},
    year = {2022}
}""",

    "mace_mp_2023": """@misc{mace_mp_2023,
    author = {Batatia, Ilyes and Benber, Philipp and Chiang, Yuan and Elena, Alin M. and others},
    title = {{A foundation model for atomistic materials chemistry}},
    year = {2023},
    eprint = {2401.00096},
    archiveprefix = {arXiv},
    primaryclass = {cond-mat.mtrl-sci}
}""",

    "mace_mpa_2024": """@misc{mace_mpa_2024,
    author = {{MACE Developers}},
    title = {{MACE-MPA-0: Matbench Discovery State-of-the-Art Foundation Model}},
    year = {2024},
    url = {https://github.com/ACEsuit/mace},
    note = {Trained on MPtrj + Alexandria + sAlex datasets, PBE+U level}
}""",

    "mace_omat_2024": """@misc{mace_omat_2024,
    author = {{MACE Developers}},
    title = {{MACE-OMAT-0: Foundation Model Optimized for Phonon Calculations}},
    year = {2024},
    url = {https://github.com/ACEsuit/mace},
    note = {Trained on OMAT dataset, PBE level, best accuracy for phonons}
}""",

    "mace_matpes_2024": """@misc{mace_matpes_2024,
    author = {{MACE Developers}},
    title = {{MACE-MATPES: Foundation Models with r2SCAN Functional}},
    year = {2024},
    url = {https://github.com/ACEsuit/mace-foundations},
    note = {MATPES dataset with r2SCAN meta-GGA DFT}
}""",

    "mace_mh_2024": """@misc{mace_mh_2024,
    author = {{MACE Developers}},
    title = {{MACE-MH: Multi-Head Foundation Models for Materials Chemistry}},
    year = {2024},
    url = {https://huggingface.co/mace-foundations/mace-mh-1},
    note = {Multi-head model trained on OMAT/OMOL/OC20/MATPES}
}""",

    # NequIP references
    "nequip2022": """@article{nequip2022,
    author = {Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P. and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E. and Kozinsky, Boris},
    title = {{E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials}},
    journal = {Nature Communications},
    volume = {13},
    pages = {2453},
    year = {2022},
    doi = {10.1038/s41467-022-29939-5}
}""",

    "allegro2023": """@article{allegro2023,
    author = {Musaelian, Albert and Batzner, Simon and Jober, Anders and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P. and Kozinsky, Boris},
    title = {{Learning local equivariant representations for scalable molecular dynamics}},
    journal = {Nature Communications},
    volume = {14},
    pages = {579},
    year = {2023},
    doi = {10.1038/s41467-023-36329-y}
}""",
}

# Publication-style references (for PDF report)
PUBLICATION_REFERENCES = {
    "orb2024": "Orbital Materials. ORB Models: Universal Neural Network Potentials for Atomistic Simulations. GitHub, 2024. https://github.com/orbital-materials/orb-models",

    "omat24_2024": "Barroso-Luque, L. et al. Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models. arXiv:2410.12771 (2024).",

    "phonopy2015": "Togo, A. & Tanaka, I. First principles phonon calculations in materials science. Scr. Mater. 108, 1–5 (2015). https://doi.org/10.1016/j.scriptamat.2015.07.021",

    "qha_togo2010": "Togo, A., Chaput, L., Tanaka, I. & Hug, G. First-principles phonon calculations of thermal expansion in Ti<sub>3</sub>SiC<sub>2</sub>, Ti<sub>3</sub>AlC<sub>2</sub>, and Ti<sub>3</sub>GeC<sub>2</sub>. Phys. Rev. B 81, 174301 (2010). https://doi.org/10.1103/PhysRevB.81.174301",

    "phono3py2015": "Togo, A., Chaput, L. & Tanaka, I. Distributions of phonon lifetimes in Brillouin zones. Phys. Rev. B 91, 094306 (2015). https://doi.org/10.1103/PhysRevB.91.094306",

    "elastic_fd_2002": "Le Page, Y. & Saxe, P. Symmetry-general least-squares extraction of elastic data for strained materials from ab initio calculations of stress. Phys. Rev. B 65, 104104 (2002). https://doi.org/10.1103/PhysRevB.65.104104",

    "ptm_larsen2016": "Larsen, P. M., Schmidt, S. & Schiøtz, J. Robust structural identification via polyhedral template matching. Model. Simul. Mater. Sci. Eng. 24, 055007 (2016). https://doi.org/10.1088/0965-0393/24/5/055007",

    "elate_gaillac2016": "Gaillac, R., Pullumbi, P. & Coudert, F.-X. ELATE: an open-source online application for analysis and visualization of elastic tensors. J. Phys.: Condens. Matter 28, 275201 (2016). https://doi.org/10.1088/0953-8984/28/27/275201",

    "ase2017": "Larsen, A. H. et al. The atomic simulation environment—a Python library for working with atoms. J. Phys.: Condens. Matter 29, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e",

    "ovito2010": "Stukowski, A. Visualization and analysis of atomistic simulation data with OVITO–the Open Visualization Tool. Model. Simul. Mater. Sci. Eng. 18, 015012 (2010). https://doi.org/10.1088/0965-0393/18/1/015012",

    "sqs_zunger1990": "Zunger, A., Wei, S.-H., Ferreira, L. G. & Bernard, J. E. Special quasirandom structures. Phys. Rev. Lett. 65, 353 (1990). https://doi.org/10.1103/PhysRevLett.65.353",

    "sqsgenerator2023": "Gehringer, D., Friák, M. & Holec, D. Models of configurationally-complex alloys made simple. Comput. Phys. Commun. 286, 108664 (2023). https://doi.org/10.1016/j.cpc.2023.108664",

    "fire_optimizer2006": "Bitzek, E., Koskinen, P., Gähler, F., Moseler, M. & Gumbsch, P. Structural relaxation made simple. Phys. Rev. Lett. 97, 170201 (2006). https://doi.org/10.1103/PhysRevLett.97.170201",

    "mechelastic2021": "Singh, S., Valencia-Jaime, I., Pavlic, O. & Romero, A. H. MechElastic: A Python library for analysis of mechanical and elastic properties of bulk and 2D materials. Comput. Phys. Commun. 267, 108068 (2021). https://doi.org/10.1016/j.cpc.2021.108068",

    "voigt1928": "Voigt, W. Lehrbuch der Kristallphysik. (Teubner, Leipzig, 1928).",

    "reuss1929": "Reuss, A. Berechnung der Fließgrenze von Mischkristallen auf Grund der Plastizitätsbedingung für Einkristalle. Z. Angew. Math. Mech. 9, 49–58 (1929). https://doi.org/10.1002/zamm.19290090104",

    "hill1952": "Hill, R. The Elastic Behaviour of a Crystalline Aggregate. Proc. Phys. Soc. A 65, 349–354 (1952). https://doi.org/10.1088/0370-1298/65/5/307",

    "anisotropy_ranganathan2008": "Ranganathan, S. I. & Ostoja-Starzewski, M. Universal Elastic Anisotropy Index. Phys. Rev. Lett. 101, 055504 (2008). https://doi.org/10.1103/PhysRevLett.101.055504",

    # MACE references
    "mace2022": "Batatia, I. et al. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. NeurIPS 35, 11423–11436 (2022).",

    "mace_mp_2023": "Batatia, I. et al. A foundation model for atomistic materials chemistry. arXiv:2401.00096 (2023).",

    "mace_mpa_2024": "MACE Developers. MACE-MPA-0: Matbench Discovery State-of-the-Art Foundation Model. GitHub, 2024. https://github.com/ACEsuit/mace",

    "mace_omat_2024": "MACE Developers. MACE-OMAT-0: Foundation Model Optimized for Phonon Calculations. GitHub, 2024. https://github.com/ACEsuit/mace",

    "mace_matpes_2024": "MACE Developers. MACE-MATPES: Foundation Models with r2SCAN Functional. GitHub, 2024. https://github.com/ACEsuit/mace-foundations",

    "mace_mh_2024": "MACE Developers. MACE-MH: Multi-Head Foundation Models for Materials Chemistry. Hugging Face, 2024. https://huggingface.co/mace-foundations/mace-mh-1",

    # NequIP references
    "nequip2022": "Batzner, S. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat. Commun. 13, 2453 (2022). https://doi.org/10.1038/s41467-022-29939-5",

    "allegro2023": "Musaelian, A. et al. Learning local equivariant representations for scalable molecular dynamics. Nat. Commun. 14, 579 (2023). https://doi.org/10.1038/s41467-023-36329-y",
}

# Computational methods with their descriptions and references
COMPUTATIONAL_METHODS = {
    "orb_calculator": ComputationalMethod(
        name="ORB Universal Neural Network Potential",
        description="Universal equivariant neural network potential trained on the Open Materials 2024 (OMat24) dataset containing over 100 million DFT calculations. Provides near-DFT accuracy at a fraction of the computational cost.",
        parameters={
            "model_direct": "orb-v3-direct-20-omat (fast, analytical forces)",
            "model_conservative": "orb-v3-conservative-inf-omat (accurate, backprop forces)",
            "precision": "float32-high (direct) / float32-highest (conservative)",
        },
        references=["orb2024", "omat24_2024"],
    ),

    "mace_calculator": ComputationalMethod(
        name="MACE Foundation Model Universal Potentials",
        description="Higher-order equivariant message passing neural networks providing near-DFT accuracy. MACE-MPA-0 achieves state-of-the-art on Matbench Discovery benchmark (trained on MPtrj + Alexandria + sAlex, PBE+U). MACE-OMAT-0 is optimized for phonon calculations (trained on OMAT dataset, PBE).",
        parameters={
            "mace_mpa_0": "mace-mpa-0-medium (Matbench SOTA, recommended for general materials)",
            "mace_omat_0_small": "mace-omat-0-small (fast, best for phonons)",
            "mace_omat_0_medium": "mace-omat-0-medium (accurate, best for phonons)",
            "elements": "89",
            "precision": "float32",
        },
        references=["mace2022", "mace_mp_2023", "mace_mpa_2024", "mace_omat_2024"],
    ),

    "nequip_calculator": ComputationalMethod(
        name="NequIP Equivariant Neural Network Potential",
        description="E(3)-equivariant neural network potential with state-of-the-art accuracy. OAM models trained on OMat24, MPtrj, and sAlex datasets (112.8M structures). MP models trained on MPtrj only (1.58M structures).",
        parameters={
            "model_oam_l": "nequip-oam-l (9.6M params, F1=0.893)",
            "model_oam_xl": "nequip-oam-xl (~15M params, F1=0.906, highest accuracy)",
            "model_mp_l": "nequip-mp-l (9.6M params, MPtrj only)",
            "precision": "float32",
        },
        references=["nequip2022", "allegro2023"],
    ),

    # Planned MACE foundation models (not yet implemented)
    "mace_matpes_r2scan_calculator": ComputationalMethod(
        name="MACE-MATPES r2SCAN Universal Potential",
        description="MACE foundation model trained on MATPES dataset with r2SCAN meta-GGA functional. Provides higher accuracy than PBE for materials properties, particularly for geometries and energetics.",
        parameters={
            "model": "mace-matpes-r2scan-0",
            "elements": "89",
            "dft_functional": "r2SCAN (meta-GGA)",
            "precision": "float32",
            "min_version": "mace>=0.3.10",
        },
        references=["mace2022", "mace_matpes_2024"],
    ),

    "mace_mh_calculator": ComputationalMethod(
        name="MACE-MH Multi-Head Universal Potential",
        description="Multi-head MACE foundation model pre-trained on OMAT-24 (100M inorganic crystals) and fine-tuned on diverse datasets (OMOL, OC20, MATPES). Excellent cross-domain performance for crystals, molecules, and surfaces.",
        parameters={
            "model_mh0": "mace-mh-0 (linear interaction blocks)",
            "model_mh1": "mace-mh-1 (non-linear interaction blocks, recommended)",
            "elements": "89",
            "dft_functionals": "PBE/r2SCAN/wB97M-VV10 (mixed)",
            "architecture": "512 node channels, 128 edge channels, L=1, max_ell=3",
            "precision": "float32",
            "min_version": "mace>=0.3.14",
        },
        references=["mace2022", "mace_mh_2024"],
    ),

    "structure_relaxation": ComputationalMethod(
        name="Structure Relaxation",
        description="Atomic position and cell optimization using the FIRE (Fast Inertial Relaxation Engine) algorithm with FrechetCellFilter for hydrostatic strain relaxation. Alloy generation uses a two-stage strategy: a coarse GPU pass followed by a fine CPU pass.",
        parameters={
            "optimizer": "FIRE",
            "cell_filter": "FrechetCellFilter (hydrostatic strain, isotropic cell)",
            "stage_1": "Coarse relax on GPU — FIRE, fmax=0.01 eV/Å, max_steps=500",
            "stage_2": "Fine relax on CPU — FIRE, fmax=0.001 eV/Å (user-tunable), max_steps=500",
            "target_supercell_size": "48 atoms (session default; user-configurable)",
            "fmax_default": "0.005 eV/Å (generic relaxation helper)",
            "max_steps_default": "500",
        },
        references=["fire_optimizer2006", "ase2017"],
    ),

    "sqs_generation": ComputationalMethod(
        name="Special Quasirandom Structure (SQS)",
        description="Generates special quasirandom structures that best approximate the correlation functions of a random alloy. Minimizes short-range order to represent disordered solid solutions.",
        parameters={
            "method": "sqsgenerator library",
            "target": "Minimize pair correlation deviation",
            "iterations": "1,000,000 Monte Carlo swaps (tool default; library default is 10,000,000)",
            "shell_weights": "{1: 1.0, 2: 0.5} — NN dominant, 2nd NN half-weighted",
        },
        references=["sqs_zunger1990", "sqsgenerator2023"],
    ),

    "elastic_finite_diff": ComputationalMethod(
        name="Finite Difference Elastic Constants",
        description="Calculates the full 6×6 elastic stiffness tensor (relaxed/Born tensor) using a finite-strain energy method. Applies 180 asymmetric strain deformations to the relaxed structure, relaxes atomic positions at fixed cell, and recovers the symmetric tensor by least-squares regression of the quadratic energy-strain relationship (the large number of deformations averages out noise and captures off-diagonal couplings).",
        parameters={
            "strain_magnitude": "0.01 (1%, default)",
            "minimum_allowed": "0.01 (enforced; smaller values become too noisy vs. fmax=0.005 relaxation)",
            "deformation_types": "180 (all asymmetric strain combinations; least-squares recovery of symmetric C tensor)",
            "inner_relaxation": "FIRE, fmax=0.005 eV/Å, max_steps=100, cell fixed (hydrostatic_strain=False)",
            "fitting": "Least-squares regression of energy vs. strain over all 180 deformations",
            "tensor_type": "Relaxed (Born) stiffness tensor — matches experiment",
            "stress_calculation": "Analytical from ML potential",
            "units": "GPa, Voigt 6×6 form",
        },
        references=["elastic_fd_2002", "mechelastic2021"],
    ),

    "qha": ComputationalMethod(
        name="Quasi-Harmonic Approximation (QHA)",
        description="Temperature-dependent thermodynamic properties via phonon calculations at multiple volumes. Captures thermal expansion and temperature-dependent bulk modulus through volume-dependent phonon frequencies.",
        parameters={
            "num_volumes": "11 (default)",
            "strain_range": "±10% volumetric (default)",
            "phonon_mesh": "20×20×20 (default)",
            "temperature_range": "0–600 K in 10 K steps",
            "phonopy_displacement": "0.01 Å (FC2 finite difference)",
            "symprec": "5×10<super>-3</super> Å",
            "primitive_matrix": "'auto' (automatic primitive cell detection)",
            "supercell_matrix": "identity (default — uses relaxed supercell as-is)",
            "per_volume_relaxation": "FIRE, fmax=0.01 eV/Å, max_steps=100, cell fixed",
        },
        references=["phonopy2015", "qha_togo2010"],
    ),

    "thermal_conductivity": ComputationalMethod(
        name="Lattice Thermal Conductivity",
        description="Calculates lattice thermal conductivity using third-order force constants and the Boltzmann transport equation under the relaxation time approximation (RTA). Accounts for three-phonon scattering processes.",
        parameters={
            "method": "phono3py",
            "transport_method": "RTA (relaxation time approximation)",
            "fc3_displacement": "0.01 Å",
            "mesh": "20×20×20 (default)",
            "temperature_range": "0–610 K in 10 K steps",
            "symprec": "5×10<super>-3</super> Å",
            "supercell_matrix": "identity (default)",
            "cutoff_pair_distance": "None (default — include all three-phonon interactions)",
            "scattering": "Three-phonon processes",
            "output_file": "thermal_conductivity/kappa-m{nx}{ny}{nz}.hdf5",
        },
        references=["phono3py2015"],
    ),

    "ptm_analysis": ComputationalMethod(
        name="Polyhedral Template Matching (PTM)",
        description="Local atomic environment classification algorithm that identifies crystal structure types (FCC, BCC, HCP, etc.) by matching to ideal polyhedral templates. Robust to thermal vibrations and defects.",
        parameters={
            "rmsd_cutoff": "0.0 (OVITO convention: 0 disables the cutoff — every atom assigned to its best-matching template)",
            "structure_types": "FCC, BCC, HCP, ICO, SC, cubic diamond (OVITO default set + explicitly-enabled ICO/SC/diamond)",
        },
        references=["ptm_larsen2016", "ovito2010"],
    ),

    "elate_analysis": ComputationalMethod(
        name="ELATE Elastic Anisotropy Analysis",
        description="Comprehensive elastic anisotropy analysis including Voigt-Reuss-Hill averaging, directional Young's modulus, linear compressibility, shear modulus, and Poisson's ratio visualization.",
        parameters={
            "averaging": "Voigt, Reuss, Hill",
            "properties": "E, β, G, ν (directional)",
            "visualization": "2D polar plots, 3D surfaces",
        },
        references=["elate_gaillac2016", "mechelastic2021"],
    ),

    "formation_energy": ComputationalMethod(
        name="Formation Energy Calculation",
        description="Calculates the formation energy of an alloy relative to its constituent elements. Uses ground-state energies of pure elements as reference.",
        parameters={
            "reference_mode": "ground_state (minimum energy structure)",
            "formula": "E_form = E_alloy - Σ(x_i × E_ref_i)",
            "reference_structures": "sc, bcc, fcc, hcp, diamond — lowest-energy selected per element",
        },
        references=["ase2017"],
    ),

    "rdf_analysis": ComputationalMethod(
        name="Radial Distribution Function (RDF)",
        description="Radial distribution function computed via OVITO's coordination analysis. Partial RDFs are computed per element pair and combined as g_total(r) = Σ c_i·c_j·g_ij(r).",
        parameters={
            "method": "OVITO CoordinationAnalysisModifier",
            "cutoff": "10.0 Å (default; user-tunable)",
            "n_bins": "200",
            "partial": "True (per-element-pair + total)",
        },
        references=["ovito2010"],
    ),
}


def get_methods_for_calculation(
    calculation_type: str,
    calculator_name: str = None
) -> List[ComputationalMethod]:
    """Get computational methods used for a specific calculation type.

    Args:
        calculation_type: One of 'alloy_generation', 'elastic', 'qha', 'thermal_conductivity'
        calculator_name: Calculator used (e.g., 'orb-v3-direct-20-omat', 'mace-mpa-0-medium', 'nequip-oam-l').
                        If None, defaults to ORB calculator.

    Returns:
        List of ComputationalMethod objects
    """
    # Determine calculator method key based on calculator name
    if calculator_name:
        if calculator_name.startswith('mace-'):
            # Route to specific MACE calculator documentation
            if 'matpes' in calculator_name or 'r2scan' in calculator_name.lower():
                calc_key = "mace_matpes_r2scan_calculator"
            elif 'mh' in calculator_name:
                calc_key = "mace_mh_calculator"
            elif 'mpa' in calculator_name or 'omat' in calculator_name:
                calc_key = "mace_calculator"  # MACE-MPA-0 and MACE-OMAT-0 (current)
            else:
                calc_key = "mace_calculator"  # Default MACE models
        elif calculator_name.startswith('nequip-'):
            calc_key = "nequip_calculator"
        else:
            calc_key = "orb_calculator"
    else:
        calc_key = "orb_calculator"

    method_mapping = {
        "alloy_generation": [
            calc_key,
            "sqs_generation",
            "structure_relaxation",
            "ptm_analysis",
            "rdf_analysis",
            "formation_energy",
        ],
        "elastic": [
            calc_key,
            "elastic_finite_diff",
            "elate_analysis",
        ],
        "qha": [
            calc_key,
            "qha",
        ],
        "thermal_conductivity": [
            calc_key,
            "thermal_conductivity",
        ],
    }

    method_keys = method_mapping.get(calculation_type, [])
    return [COMPUTATIONAL_METHODS[key] for key in method_keys if key in COMPUTATIONAL_METHODS]


def get_all_references_for_calculation(calculation_type: str) -> List[str]:
    """Get all reference keys for a specific calculation type.

    Args:
        calculation_type: One of 'alloy_generation', 'elastic', 'qha', 'thermal_conductivity'

    Returns:
        List of reference keys (deduplicated)
    """
    methods = get_methods_for_calculation(calculation_type)
    all_refs = []
    for method in methods:
        all_refs.extend(method.references)
    # Remove duplicates while preserving order
    seen = set()
    unique_refs = []
    for ref in all_refs:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    return unique_refs


def generate_bibtex_file(reference_keys: List[str], filepath: Path) -> None:
    """Generate a BibTeX file with selected references.

    Args:
        reference_keys: List of reference keys to include
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("% BibTeX references for OptiMat Alloys calculations\n")
        f.write("% Generated automatically\n\n")

        for key in reference_keys:
            if key in BIBTEX_REFERENCES:
                f.write(BIBTEX_REFERENCES[key])
                f.write("\n\n")


def get_publication_references(reference_keys: List[str]) -> List[str]:
    """Get publication-style reference strings.

    Args:
        reference_keys: List of reference keys

    Returns:
        List of formatted reference strings
    """
    refs = []
    for i, key in enumerate(reference_keys, 1):
        if key in PUBLICATION_REFERENCES:
            refs.append(f"[{i}] {PUBLICATION_REFERENCES[key]}")
    return refs


def format_inline_citations(
    method_refs: List[str],
    all_ref_keys: List[str]
) -> str:
    """Format inline citation numbers for a method's references.

    Args:
        method_refs: List of reference keys for this method
        all_ref_keys: Master list of all reference keys (determines numbering)

    Returns:
        Formatted string like "[1, 3, 5]" or "" if no valid refs
    """
    ref_numbers = []
    for ref_key in method_refs:
        if ref_key in all_ref_keys:
            # Reference numbers are 1-indexed
            ref_num = all_ref_keys.index(ref_key) + 1
            ref_numbers.append(ref_num)

    if not ref_numbers:
        return ""

    # Sort and format
    ref_numbers.sort()
    return "[" + ", ".join(str(n) for n in ref_numbers) + "]"


# Software versions (can be updated dynamically)
def get_software_versions() -> Dict[str, str]:
    """Get versions of key software packages.

    Returns:
        Dictionary of package names to version strings
    """
    versions = {}

    try:
        import ase
        versions["ASE"] = ase.__version__
    except (ImportError, AttributeError):
        versions["ASE"] = "unknown"

    try:
        import phonopy
        versions["Phonopy"] = phonopy.__version__
    except (ImportError, AttributeError):
        versions["Phonopy"] = "not installed"

    try:
        import ovito
        versions["OVITO"] = ovito.version_string
    except (ImportError, AttributeError):
        versions["OVITO"] = "unknown"

    try:
        import torch
        versions["PyTorch"] = torch.__version__
    except (ImportError, AttributeError):
        versions["PyTorch"] = "unknown"

    try:
        import numpy
        versions["NumPy"] = numpy.__version__
    except (ImportError, AttributeError):
        versions["NumPy"] = "unknown"

    try:
        import orb_models
        versions["ORB Models"] = getattr(orb_models, "__version__", "installed")
    except (ImportError, AttributeError):
        versions["ORB Models"] = "unknown"

    try:
        import mace
        versions["MACE"] = getattr(mace, "__version__", "installed")
    except (ImportError, AttributeError):
        versions["MACE"] = "not installed"

    try:
        import nequip
        versions["NequIP"] = getattr(nequip, "__version__", "installed")
    except (ImportError, AttributeError):
        try:
            import fairchem
            versions["NequIP"] = f"via fairchem {getattr(fairchem, '__version__', 'installed')}"
        except (ImportError, AttributeError):
            versions["NequIP"] = "not installed"

    try:
        import phono3py
        versions["phono3py"] = getattr(phono3py, "__version__", "installed")
    except (ImportError, AttributeError):
        versions["phono3py"] = "not installed"

    try:
        import sqsgenerator
        versions["sqsgenerator"] = getattr(sqsgenerator, "__version__", "installed")
    except (ImportError, AttributeError):
        versions["sqsgenerator"] = "not installed"

    try:
        import mechelastic
        versions["MechElastic"] = getattr(mechelastic, "__version__", "installed")
    except (ImportError, AttributeError):
        versions["MechElastic"] = "not installed"

    return versions
