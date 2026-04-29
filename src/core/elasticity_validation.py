"""
Validation utilities for elastic stiffness tensors.

This module provides functions to validate elastic tensors for mechanical stability,
numerical conditioning, and proper symmetry. Validation is critical before using
ELATE analysis, as invalid tensors cause numerical failures and incorrect results.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from ase import Atoms


class ElasticTensorValidationResult:
    """
    Result of elastic tensor validation.

    Attributes
    ----------
    is_valid : bool
        Overall validation status
    is_positive_definite : bool
        Whether all eigenvalues are positive (Born stability criterion)
    is_symmetric : bool
        Whether tensor is symmetric within tolerance
    eigenvalues : np.ndarray
        Eigenvalues of the tensor (sorted ascending)
    condition_number : float
        Ratio of largest to smallest eigenvalue (measures numerical stability)
    symmetry_error : float
        Maximum absolute difference between Cij and Cji
    warnings : list[str]
        Non-fatal issues (e.g., high condition number, near-incompressibility)
    errors : list[str]
        Fatal issues (e.g., negative eigenvalues, non-symmetric)
    """

    def __init__(self):
        self.is_valid = True
        self.is_positive_definite = True
        self.is_symmetric = True
        self.eigenvalues = None
        self.condition_number = 0.0
        self.symmetry_error = 0.0
        self.warnings = []
        self.errors = []

    def add_warning(self, message: str):
        """Add a non-fatal warning."""
        self.warnings.append(message)

    def add_error(self, message: str):
        """Add a fatal error and mark validation as failed."""
        self.errors.append(message)
        self.is_valid = False


def validate_elastic_tensor(
    C_voigt: np.ndarray,
    symmetry_tol: float = 1e-3,
    eigenvalue_tol: float = 1e-6,
    condition_threshold: float = 1e6
) -> ElasticTensorValidationResult:
    """
    Validate elastic stiffness tensor for mechanical stability and numerical quality.

    Performs comprehensive checks:
    1. **Born stability criterion**: All eigenvalues must be positive (mechanical stability)
    2. **Symmetry**: Cij ≈ Cji (physical requirement)
    3. **Conditioning**: Ratio of largest/smallest eigenvalue (numerical stability)
    4. **Near-incompressibility**: Warns if Poisson ratio approaches 0.5

    Parameters
    ----------
    C_voigt : np.ndarray
        6x6 elastic stiffness tensor in Voigt notation (GPa)
    symmetry_tol : float
        Maximum allowed asymmetry |Cij - Cji| (default: 1e-3 GPa)
    eigenvalue_tol : float
        Minimum eigenvalue to consider positive (default: 1e-6 GPa)
    condition_threshold : float
        Condition number above which to warn (default: 1e6)

    Returns
    -------
    result : ElasticTensorValidationResult
        Comprehensive validation result with diagnostics

    Examples
    --------
    >>> C = np.eye(6) * 100  # Simple isotropic-like tensor
    >>> result = validate_elastic_tensor(C)
    >>> result.is_valid
    True
    >>> result.is_positive_definite
    True

    Notes
    -----
    - Negative eigenvalues indicate mechanically unstable material (imaginary phonon modes)
    - High condition number (>1e6) indicates numerical instability in downstream calculations
    - Near-incompressible materials (ν→0.5) may cause issues in ELATE analysis

    References
    ----------
    Born stability criterion:
    - M. Born and K. Huang, "Dynamical Theory of Crystal Lattices" (1954)
    - https://en.wikipedia.org/wiki/Born_stability_criterion
    """
    result = ElasticTensorValidationResult()

    # Validate input shape
    if C_voigt.shape != (6, 6):
        result.add_error(f"Expected 6x6 tensor, got shape {C_voigt.shape}")
        return result

    # Check symmetry (Cij should equal Cji)
    symmetry_matrix = C_voigt - C_voigt.T
    max_asymmetry = np.abs(symmetry_matrix).max()
    result.symmetry_error = float(max_asymmetry)

    if max_asymmetry > symmetry_tol:
        result.is_symmetric = False
        result.add_error(
            f"Tensor is not symmetric (max |Cij - Cji| = {max_asymmetry:.4f} GPa > {symmetry_tol:.4f} GPa). "
            f"Physical elastic tensors must be symmetric."
        )

    # Compute eigenvalues (measures for positive definiteness and conditioning)
    try:
        eigenvalues = np.linalg.eigvalsh(C_voigt)  # eigvalsh assumes symmetric
        result.eigenvalues = eigenvalues
    except np.linalg.LinAlgError as e:
        result.add_error(f"Failed to compute eigenvalues: {e}")
        return result

    # Check Born stability criterion (all eigenvalues must be positive)
    negative_eigs = eigenvalues[eigenvalues < -eigenvalue_tol]
    near_zero_eigs = eigenvalues[(eigenvalues >= -eigenvalue_tol) & (eigenvalues < eigenvalue_tol)]

    if len(negative_eigs) > 0:
        result.is_positive_definite = False
        result.add_error(
            f"Tensor violates Born stability criterion: {len(negative_eigs)} negative eigenvalue(s). "
            f"Negative eigenvalues: {negative_eigs}. "
            f"This indicates a mechanically unstable material (imaginary phonon modes)."
        )

    if len(near_zero_eigs) > 0:
        result.add_warning(
            f"Tensor has {len(near_zero_eigs)} near-zero eigenvalue(s) "
            f"({near_zero_eigs[0]:.2e} GPa). This may indicate numerical issues."
        )

    # Check condition number (ratio of largest to smallest eigenvalue)
    if len(eigenvalues) > 0 and eigenvalues[0] > eigenvalue_tol:
        condition_number = abs(eigenvalues[-1] / eigenvalues[0])
        result.condition_number = float(condition_number)

        if condition_number > condition_threshold:
            result.add_warning(
                f"High condition number ({condition_number:.2e}). "
                f"Tensor is ill-conditioned, which may cause numerical instability "
                f"in downstream calculations (e.g., ELATE analysis)."
            )

    # Check for near-incompressibility (Poisson ratio approaching 0.5)
    # Estimate from Voigt averaging
    C11, C22, C33 = C_voigt[0, 0], C_voigt[1, 1], C_voigt[2, 2]
    C12, C13, C23 = C_voigt[0, 1], C_voigt[0, 2], C_voigt[1, 2]
    C44, C55, C66 = C_voigt[3, 3], C_voigt[4, 4], C_voigt[5, 5]

    K = (C11 + C22 + C33 + 2 * (C12 + C13 + C23)) / 9.0
    G = ((C11 + C22 + C33) - (C12 + C13 + C23) + 3 * (C44 + C55 + C66)) / 15.0

    if G > 1e-3:  # Avoid division by zero
        E = 9 * K * G / (3 * K + G)
        nu = (3 * K - 2 * G) / (6 * K + 2 * G)

        if nu > 0.49:
            result.add_warning(
                f"Near-incompressible material detected (Poisson ratio ν = {nu:.4f} ≈ 0.5). "
                f"This may cause numerical issues in ELATE analysis. "
                f"Consider using ν = 0.495 if problems occur."
            )

    return result


def diagnose_invalid_tensor(
    C_voigt: np.ndarray,
    atoms: Optional[Atoms] = None,
    epsilon: float = 0.01,
    max_force: Optional[float] = None
) -> Dict[str, any]:
    """
    Diagnose root causes of invalid elastic tensor and suggest fixes.

    Analyzes common failure modes:
    1. Insufficient structural relaxation (residual forces)
    2. Strain magnitude too small (numerical noise)
    3. Intrinsically unstable structure

    Parameters
    ----------
    C_voigt : np.ndarray
        6x6 elastic stiffness tensor that failed validation
    atoms : Atoms, optional
        The structure used for calculation (for force analysis)
    epsilon : float
        Strain magnitude used in calculation
    max_force : float, optional
        Maximum force in structure (eV/Å), if known

    Returns
    -------
    diagnostics : dict
        Dictionary with keys:
        - 'likely_causes': List of probable root causes
        - 'recommendations': List of suggested fixes
        - 'severity': 'critical', 'high', or 'moderate'

    Examples
    --------
    >>> result = validate_elastic_tensor(C)
    >>> if not result.is_valid:
    ...     diag = diagnose_invalid_tensor(C, atoms, epsilon=0.01, max_force=0.008)
    ...     print("Likely causes:", diag['likely_causes'])
    ...     print("Try:", diag['recommendations'])
    """
    likely_causes = []
    recommendations = []
    severity = 'moderate'

    # Check residual forces
    if max_force is not None:
        if max_force > 0.005:
            likely_causes.append(
                f"Structure not fully relaxed (max force = {max_force:.4f} eV/Å > 0.005 eV/Å)"
            )
            recommendations.append(
                "Re-relax structure with stricter convergence: fmax=0.001 eV/Å or lower"
            )
            severity = 'high'
        elif max_force > 0.002:
            likely_causes.append(
                f"Structure has non-negligible forces ({max_force:.4f} eV/Å)"
            )
            recommendations.append(
                "Try re-relaxing with fmax=0.001 eV/Å for better accuracy"
            )

    # Check strain magnitude
    if epsilon < 0.01:
        likely_causes.append(
            f"Strain magnitude may be too small (ε = {epsilon:.3f} = {epsilon*100:.1f}%)"
        )
        recommendations.append(
            "Increase strain magnitude: try epsilon=0.01 (1%) or epsilon=0.02 (2%)"
        )

    # Check for extreme anisotropy
    result = validate_elastic_tensor(C_voigt)
    if result.eigenvalues is not None:
        eigs = result.eigenvalues
        if len(eigs) > 0 and eigs[-1] > 0:
            # Check ratio of largest positive to most negative
            if len(eigs[eigs < 0]) > 0:
                ratio = abs(eigs[-1] / eigs[0])
                if ratio > 100:
                    likely_causes.append(
                        f"Extreme anisotropy detected (eigenvalue ratio: {ratio:.1f})"
                    )
                    recommendations.append(
                        "Structure may be intrinsically unstable. "
                        "Check if structure collapsed or transformed during deformation."
                    )
                    severity = 'critical'

    # General recommendations
    if not recommendations:
        recommendations.append(
            "Re-calculate elastic tensor with different settings (optimizer, epsilon)"
        )
        recommendations.append(
            "Verify structure is in a local energy minimum (check energy vs. known phases)"
        )
        recommendations.append(
            "Consider using a different calculator or tighter convergence criteria"
        )

    return {
        'likely_causes': likely_causes,
        'recommendations': recommendations,
        'severity': severity
    }


def validate_elate_results(properties: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate ELATE analysis results for correctness.

    Detects silent failures in ELATE calculations that produce physically
    impossible or numerically invalid results.

    Parameters
    ----------
    properties : dict
        Properties dictionary from ElasticAnisotropyAnalyzer.compute_comprehensive_properties()

    Returns
    -------
    is_valid : bool
        True if results are physically reasonable
    error_message : str or None
        Description of problem if invalid, None if valid

    Examples
    --------
    >>> props = analyzer.compute_comprehensive_properties()
    >>> is_valid, error = validate_elate_results(props)
    >>> if not is_valid:
    ...     print(f"ELATE failed: {error}")

    Notes
    -----
    Common failure indicators:
    - Universal anisotropy index exactly 0.0 (impossible for real materials)
    - Negative Pugh ratio (K/G must be positive)
    - All directional ranges are [0, 0] (ELATE couldn't compute)
    - NaN or infinity values
    """
    # Check for NaN or infinity
    for key, value in properties.items():
        if isinstance(value, (int, float)):
            if np.isnan(value):
                return False, f"ELATE produced NaN value for {key}"
            # Note: infinity is valid for some properties (e.g., AU for highly anisotropic)

    # Check universal anisotropy index
    AU = properties.get('universal_anisotropy_index', 0.0)
    if AU == 0.0 and not np.isinf(AU):
        # AU = 0 only for perfectly isotropic materials, very rare in practice
        # If other indicators of failure present, this is suspicious
        pass  # Will check other indicators below

    # Check Pugh ratio (K/G must be positive)
    pugh = properties.get('pugh_ratio_hill', 0.0)
    if pugh < 0:
        return False, f"ELATE produced invalid Pugh ratio ({pugh:.2f}). Pugh ratio (K/G) must be positive."

    # Check directional ranges (should not all be zero)
    ranges_zero = 0
    for key in ['min_youngs_modulus_GPa', 'max_youngs_modulus_GPa',
                'min_poisson_ratio', 'max_poisson_ratio',
                'min_shear_modulus_GPa', 'max_shear_modulus_GPa']:
        if abs(properties.get(key, 0.0)) < 1e-12:
            ranges_zero += 1

    if ranges_zero >= 4:  # More than half of directional properties are zero
        return False, "ELATE failed to compute directional properties (all ranges are zero)"

    # Check that min < max for directional properties
    if properties.get('min_youngs_modulus_GPa', 0) >= properties.get('max_youngs_modulus_GPa', 1):
        if properties.get('max_youngs_modulus_GPa', 0) > 0:  # Exclude zero case
            return False, "ELATE produced invalid Young's modulus range (min >= max)"

    return True, None


def format_validation_message(
    result: ElasticTensorValidationResult,
    diagnostics: Optional[Dict] = None
) -> str:
    """
    Format validation result as user-friendly markdown message.

    Parameters
    ----------
    result : ElasticTensorValidationResult
        Validation result
    diagnostics : dict, optional
        Diagnostics from diagnose_invalid_tensor()

    Returns
    -------
    message : str
        Formatted markdown message
    """
    if result.is_valid:
        msg = "✅ **Elastic tensor validated successfully**\n\n"
        msg += f"- Born stability: ✓ (all eigenvalues positive)\n"
        msg += f"- Symmetry: ✓ (max error {result.symmetry_error:.2e} GPa)\n"
        msg += f"- Condition number: {result.condition_number:.2e}\n"

        if result.warnings:
            msg += "\n**Warnings:**\n"
            for w in result.warnings:
                msg += f"- ⚠️ {w}\n"

        return msg
    else:
        msg = "❌ **Elastic tensor validation FAILED**\n\n"

        # Show errors
        if result.errors:
            msg += "**Critical issues:**\n"
            for e in result.errors:
                msg += f"- {e}\n"
            msg += "\n"

        # Show eigenvalues if available
        if result.eigenvalues is not None:
            eigs = result.eigenvalues
            msg += "**Eigenvalue spectrum:**\n"
            msg += "```\n"
            for i, eig in enumerate(eigs):
                status = "❌ NEGATIVE" if eig < 0 else "✓"
                msg += f"λ{i+1} = {eig:>10.4f} GPa  {status}\n"
            msg += "```\n\n"

        # Show diagnostics
        if diagnostics:
            if diagnostics.get('likely_causes'):
                msg += "**Likely causes:**\n"
                for cause in diagnostics['likely_causes']:
                    msg += f"- {cause}\n"
                msg += "\n"

            if diagnostics.get('recommendations'):
                msg += "**Recommendations:**\n"
                for rec in diagnostics['recommendations']:
                    msg += f"- {rec}\n"
                msg += "\n"

        msg += "_Elastic tensor saved for debugging, but ELATE analysis will be skipped._"

        return msg
