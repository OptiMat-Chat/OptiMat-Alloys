"""
OVITO-based structure rendering.

This module provides functions for rendering atomic structures
using OVITO's high-quality ray tracing.
"""

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
warnings.filterwarnings('ignore', category=FutureWarning, module='ovito.io.ase')

from pathlib import Path
from typing import Union, Optional
from ase import Atoms

from src.storage.global_database import GlobalStructureDatabase


def render_structure(
    atoms: Atoms,
    structure_uuid: str,
    filename: str = "structure_analysis.png",
    db: Optional[GlobalStructureDatabase] = None
) -> str:
    """
    Render structure with PTM structural analysis coloring.

    Atoms are colored by their local crystal structure type
    (FCC=green, HCP=red, BCC=blue, etc.).

    Args:
        atoms: ASE Atoms object to render
        structure_uuid: Structure UUID (32-character hex string, determines output directory)
        filename: Output filename (default: "structure_analysis.png")
        db: Optional database instance (creates new one if None)

    Returns:
        Full path to saved image

    Examples:
        >>> path = render_structure(atoms, "a1b2c3d4...")  # doctest: +SKIP
    """
    try:
        return _render_structure_impl(atoms, structure_uuid, filename, db)
    except Exception as e:
        print(f"Error rendering structure: {e}")
        raise


def _render_structure_impl(
    atoms: Atoms,
    structure_uuid: str,
    filename: str,
    db: Optional[GlobalStructureDatabase]
) -> str:
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.vis import Viewport, TachyonRenderer, ColorLegendOverlay
    from ovito.qt_compat import QtCore
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    import ovito

    # Clear the scene's pipelines
    ovito.scene.pipelines.clear()

    # Convert ASE atoms to OVITO DataCollection
    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))

    # Add PTM modifier for structural analysis
    ptm_modifier = PolyhedralTemplateMatchingModifier()
    ptm_modifier.rmsd_cutoff = 0.0
    ptm_modifier.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
    ptm_modifier.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
    ptm_modifier.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True

    pipeline.modifiers.append(ptm_modifier)
    structures = [s for s in ptm_modifier.structures if s.enabled]

    pipeline.add_to_scene()

    # Set up viewport
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(-2, -1, -1))
    vp.zoom_all()

    # Create color legend for structure types
    legend = ColorLegendOverlay(
        title=' ',
        alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom,
        orientation=QtCore.Qt.Orientation.Vertical,
        legend_size=0.1*len(structures),
        aspect_ratio=2*len(structures),
        offset_y=0.0,
        offset_x=0.01,
        font_size=0.5/len(structures)
    )
    legend.property = 'particles/Structure Type'
    vp.overlays.append(legend)

    # Get structure directory from database
    if db is None:
        db = GlobalStructureDatabase()
    structure_dir = db.get_structure_directory(structure_uuid)
    structure_dir.mkdir(parents=True, exist_ok=True)
    image_path = structure_dir / filename

    # Render with Tachyon (high quality ray tracer)
    vp.render_image(
        filename=str(image_path),
        size=(1280, 960),
        background=(1, 1, 1),
        renderer=TachyonRenderer(ambient_occlusion=True, shadows=True)
    )

    return str(image_path)


def render_atoms(
    atoms: Atoms,
    structure_uuid: str,
    filename: str = "structure_elements.png",
    db: Optional[GlobalStructureDatabase] = None
) -> str:
    """
    Render structure with element-based coloring.

    Atoms are colored by their element type (default OVITO colors).

    Args:
        atoms: ASE Atoms object to render
        structure_uuid: Structure UUID (32-character hex string, determines output directory)
        filename: Output filename (default: "structure_elements.png")
        db: Optional database instance (creates new one if None)

    Returns:
        Full path to saved image

    Examples:
        >>> path = render_atoms(atoms, "a1b2c3d4...")  # doctest: +SKIP
    """
    try:
        return _render_atoms_impl(atoms, structure_uuid, filename, db)
    except Exception as e:
        print(f"Error rendering atoms: {e}")
        raise


def _render_atoms_impl(
    atoms: Atoms,
    structure_uuid: str,
    filename: str,
    db: Optional[GlobalStructureDatabase]
) -> str:
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.vis import Viewport, TachyonRenderer, ColorLegendOverlay
    from ovito.qt_compat import QtCore
    import ovito

    # Clear pipelines
    ovito.scene.pipelines.clear()

    # Get unique elements
    elements = set(atoms.get_chemical_symbols())

    # Convert to OVITO
    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()

    # Set up viewport
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(-2, -1, -1))
    vp.zoom_all()

    # Create element legend
    legend = ColorLegendOverlay(
        title='',
        alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop,
        orientation=QtCore.Qt.Orientation.Horizontal,
        legend_size=0.1*len(elements),
        aspect_ratio=2*len(elements),
        offset_y=0.01,
        font_size=0.5/len(elements)
    )
    legend.property = 'particles/Particle Type'
    vp.overlays.append(legend)

    # Get structure directory from database
    if db is None:
        db = GlobalStructureDatabase()
    structure_dir = db.get_structure_directory(structure_uuid)
    structure_dir.mkdir(parents=True, exist_ok=True)
    image_path = structure_dir / filename

    # Render
    vp.render_image(
        filename=str(image_path),
        size=(1280, 960),
        background=(1, 1, 1),
        renderer=TachyonRenderer(ambient_occlusion=True, shadows=True)
    )

    return str(image_path)


def render_trajectory(
    structure_uuid: str,
    traj_filename: str = "relaxation.traj",
    output_filename: str = "relaxation.gif",
    db: Optional[GlobalStructureDatabase] = None
) -> str:
    """
    Render trajectory animation from file.

    Args:
        structure_uuid: Structure UUID (32-character hex string, determines input and output directory)
        traj_filename: Trajectory filename in structure directory (default: "relaxation.traj")
        output_filename: Output animation filename (default: "relaxation.gif")
        db: Optional database instance (creates new one if None)

    Returns:
        Full path to saved animation

    Examples:
        >>> path = render_trajectory("a1b2c3d4...")  # doctest: +SKIP
    """
    try:
        return _render_trajectory_impl(structure_uuid, traj_filename, output_filename, db)
    except Exception as e:
        print(f"Error rendering trajectory: {e}")
        raise


def _render_trajectory_impl(
    structure_uuid: str,
    traj_filename: str,
    output_filename: str,
    db: Optional[GlobalStructureDatabase]
) -> str:
    from ovito.io import import_file
    from ovito.vis import Viewport, AnariRenderer, ColorLegendOverlay
    from ovito.qt_compat import QtCore
    import ovito

    # Clear pipelines
    ovito.scene.pipelines.clear()

    # Get structure directory and trajectory path
    if db is None:
        db = GlobalStructureDatabase()
    structure_dir = db.get_structure_directory(structure_uuid)
    traj_path = structure_dir / traj_filename

    # Import trajectory
    pipeline = import_file(str(traj_path))
    data = pipeline.compute()

    # Get unique elements
    elements = set(data.particles['Particle Type'])

    pipeline.add_to_scene()

    # Set up viewport
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(-2, -1, -1))
    vp.zoom_all()

    # Create legend
    legend = ColorLegendOverlay(
        title='',
        alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop,
        orientation=QtCore.Qt.Orientation.Horizontal,
        legend_size=0.1*len(elements),
        aspect_ratio=2*len(elements),
        offset_y=0.01,
        font_size=0.5/len(elements)
    )
    legend.property = 'particles/Particle Type'
    vp.overlays.append(legend)

    # Output path in structure directory
    image_path = structure_dir / output_filename

    # Render animation (every 10th frame)
    vp.render_anim(
        filename=str(image_path),
        size=(1280, 960),
        background=(1, 1, 1),
        every_nth=10,
        renderer=AnariRenderer()
    )

    return str(image_path)
