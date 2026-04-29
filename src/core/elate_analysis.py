"""
ELATE-based elastic anisotropy analysis.

This module provides comprehensive elastic property analysis including
directional dependence, anisotropy measures, and Voigt/Reuss/Hill averaging.

Integration with MechElastic ELATE package for rigorous elastic tensor analysis.
"""

from typing import Dict, Optional
import warnings
import numpy as np
from mechelastic.core import ELATE


class ElasticAnisotropyAnalyzer:
    """
    Wrapper around MechElastic ELATE for elastic anisotropy analysis.

    Provides:
    - Voigt/Reuss/Hill averaging for all elastic moduli
    - Universal Anisotropy Index and other anisotropy measures
    - Directional property variations (min/max values)
    - Auxetic behavior detection (negative Poisson's ratio)
    - Wave speed calculations (if density provided)

    Examples
    --------
    >>> import numpy as np
    >>> C_voigt = np.array([[153, 57, 57, 0, 0, 0],
    ...                     [57, 153, 57, 0, 0, 0],
    ...                     [57, 57, 153, 0, 0, 0],
    ...                     [0, 0, 0, 75, 0, 0],
    ...                     [0, 0, 0, 0, 75, 0],
    ...                     [0, 0, 0, 0, 0, 75]])
    >>> analyzer = ElasticAnisotropyAnalyzer(C_voigt, density_kg_m3=2330.0)
    >>> props = analyzer.compute_comprehensive_properties()
    >>> print(f"Universal Anisotropy: {props['universal_anisotropy_index']:.3f}")
    """

    def __init__(self, C_voigt: np.ndarray, density_kg_m3: float):
        """
        Initialize analyzer with stiffness tensor and density.

        Parameters
        ----------
        C_voigt : np.ndarray
            6x6 elastic stiffness tensor in Voigt notation (GPa)
        density_kg_m3 : float
            Material density in kg/m³
        """
        if C_voigt.shape != (6, 6):
            raise ValueError(f"Expected 6x6 tensor, got shape {C_voigt.shape}")

        self.C_voigt = C_voigt
        self.density = density_kg_m3

        # Initialize ELATE object
        # Note: ELATE expects list of lists, not numpy array
        # Suppress benign RuntimeWarnings from ELATE's internal calculations
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='mechelastic.core.elate')
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')
            self.elate = ELATE(s=C_voigt.tolist(), density=density_kg_m3)

    def _safe_float(self, value, default=0.0) -> float:
        """
        Safely convert ELATE value to float, handling infinity strings.

        ELATE sometimes returns '&infin;' for infinite values.

        Parameters
        ----------
        value : any
            Value to convert
        default : float
            Default value if conversion fails

        Returns
        -------
        float_value : float
            Converted value or np.inf for infinity strings
        """
        if isinstance(value, str):
            if value == '&infin;' or value.lower() == 'inf':
                return np.inf
            elif value == '-&infin;' or value.lower() == '-inf':
                return -np.inf
            else:
                try:
                    return float(value)
                except ValueError:
                    return default
        else:
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

    def _compute_universal_anisotropy(self, props: Dict) -> float:
        """
        Compute Universal Anisotropy Index (AU).

        AU = 5 * (GV/GR) + (KV/KR) - 6

        For isotropic materials, AU = 0.
        Higher values indicate stronger anisotropy.

        Parameters
        ----------
        props : dict
            ELATE properties dictionary

        Returns
        -------
        AU : float
            Universal Anisotropy Index
        """
        GV = props.get("shear_modulus_voigt", 1.0)
        GR = props.get("shear_modulus_reuss", 1.0)
        KV = props.get("bulk_modulus_voigt", 1.0)
        KR = props.get("bulk_modulus_reuss", 1.0)

        if GR > 0 and KR > 0:
            AU = 5.0 * (GV / GR) + (KV / KR) - 6.0
            return float(AU)
        else:
            return 0.0

    def compute_comprehensive_properties(self) -> Dict:
        """
        Compute all elastic properties and anisotropy measures.

        Returns
        -------
        properties : dict
            Complete property dictionary including:

            **Averaging Methods**:
            - voigt_bulk_modulus_GPa: Upper bound (uniform strain)
            - reuss_bulk_modulus_GPa: Lower bound (uniform stress)
            - hill_bulk_modulus_GPa: Average of Voigt and Reuss
            - (same for shear, Young's, and Poisson's moduli)

            **Anisotropy Measures**:
            - universal_anisotropy_index: 0=isotropic, >0=anisotropic
            - shear_anisotropy: Shear modulus variation ratio
            - youngs_anisotropy: Young's modulus variation ratio
            - poisson_anisotropy: Poisson ratio variation
            - bulk_anisotropy: Bulk modulus variation ratio

            **Directional Ranges**:
            - min/max_youngs_modulus_GPa: Range of E with direction
            - min/max_poisson_ratio: Range of ν with direction
            - min/max_shear_modulus_GPa: Range of G with direction

            **Special Properties**:
            - has_auxetic_behavior: True if min ν < 0
            - min/max_shear_wave_speed_m_s: Acoustic wave speeds
            - pugh_ratio_hill: Ductility indicator (K/G, >1.75 = ductile)

        Notes
        -----
        Voigt averaging assumes uniform strain (upper bound).
        Reuss averaging assumes uniform stress (lower bound).
        Hill averaging is the arithmetic mean of Voigt and Reuss.

        For polycrystalline materials, Hill average is often most accurate.
        """
        # Get full property dictionary from ELATE
        # Suppress benign RuntimeWarnings from ELATE's internal calculations
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='mechelastic.core.elate')
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')
            props = self.elate.to_dict()

        # Compute universal anisotropy index
        AU = self._compute_universal_anisotropy(props)

        # Extract and organize properties with consistent naming
        result = {
            # Bulk modulus (K) - resistance to volume change
            "voigt_bulk_modulus_GPa": self._safe_float(props.get("bulk_modulus_voigt", 0)),
            "reuss_bulk_modulus_GPa": self._safe_float(props.get("bulk_modulus_reuss", 0)),
            "hill_bulk_modulus_GPa": self._safe_float(props.get("bulk_modulus_hill", 0)),

            # Shear modulus (G) - resistance to shear deformation
            "voigt_shear_modulus_GPa": self._safe_float(props.get("shear_modulus_voigt", 0)),
            "reuss_shear_modulus_GPa": self._safe_float(props.get("shear_modulus_reuss", 0)),
            "hill_shear_modulus_GPa": self._safe_float(props.get("shear_modulus_hill", 0)),

            # Young's modulus (E) - stiffness in tension
            "voigt_youngs_modulus_GPa": self._safe_float(props.get("youngs_modulus_voigt", 0)),
            "reuss_youngs_modulus_GPa": self._safe_float(props.get("youngs_modulus_reuss", 0)),
            "hill_youngs_modulus_GPa": self._safe_float(props.get("youngs_modulus_hill", 0)),

            # Poisson's ratio (ν) - lateral strain ratio
            "voigt_poisson_ratio": self._safe_float(props.get("poisson_modulus_voigt", 0)),
            "reuss_poisson_ratio": self._safe_float(props.get("poisson_modulus_reuss", 0)),
            "hill_poisson_ratio": self._safe_float(props.get("poisson_modulus_hill", 0)),

            # Anisotropy measures
            "universal_anisotropy_index": AU,
            "shear_anisotropy": self._safe_float(props.get("shear_anisotropy", 0)),
            "youngs_anisotropy": self._safe_float(props.get("youngs_anisotropy", 0)),
            "poisson_anisotropy": self._safe_float(props.get("poisson_anisotropy", 0)),
            "bulk_anisotropy": self._safe_float(props.get("bulk_anisotropy", 0)),

            # Directional property ranges
            "min_youngs_modulus_GPa": self._safe_float(props.get("youngs_min", 0)),
            "max_youngs_modulus_GPa": self._safe_float(props.get("youngs_max", 0)),
            "min_poisson_ratio": self._safe_float(props.get("poisson_min", 0)),
            "max_poisson_ratio": self._safe_float(props.get("poisson_max", 0)),
            "min_shear_modulus_GPa": self._safe_float(props.get("shear_min", 0)),
            "max_shear_modulus_GPa": self._safe_float(props.get("shear_max", 0)),
            "min_bulk_modulus_GPa": self._safe_float(props.get("bulk_min", 0)),
            "max_bulk_modulus_GPa": self._safe_float(props.get("bulk_max", 0)),
            "min_linear_compressibility_TPa_inv": self._safe_float(props.get("linearCompression_min", 0)),
            "max_linear_compressibility_TPa_inv": self._safe_float(props.get("linearCompression_max", 0)),

            # Auxetic behavior (negative Poisson's ratio)
            "has_auxetic_behavior": self._safe_float(props.get("poisson_min", 0)) < 0,

            # Wave speeds (if density provided)
            "min_shear_wave_speed_m_s": self._safe_float(props.get("shearSpeed_min", 0)),
            "max_shear_wave_speed_m_s": self._safe_float(props.get("shearSpeed_max", 0)),
            "min_compression_wave_speed_m_s": self._safe_float(props.get("compressionSpeed_min", 0)),
            "max_compression_wave_speed_m_s": self._safe_float(props.get("compressionSpeed_max", 0)),

            # Pugh ratio (ductility indicator: K/G > 1.75 suggests ductile)
            "pugh_ratio_voigt": self._safe_float(props.get("pugh_ratio_voigt", 0)),
            "pugh_ratio_reuss": self._safe_float(props.get("pugh_ratio_reuss", 0)),
            "pugh_ratio_hill": self._safe_float(props.get("pugh_ratio_hill", 0)),
        }

        return result

    def get_elate_object(self) -> ELATE:
        """
        Get the underlying ELATE object for advanced operations.

        Returns
        -------
        elate : ELATE
            MechElastic ELATE object
        """
        return self.elate


def compute_elate_properties(
    C_voigt: np.ndarray,
    density_g_cm3: float
) -> Dict:
    """
    Convenience function for ELATE property computation.

    Parameters
    ----------
    C_voigt : np.ndarray
        6x6 stiffness tensor in Voigt notation (GPa)
    density_g_cm3 : float
        Density in g/cm³ (converted to kg/m³ internally)

    Returns
    -------
    properties : dict
        Comprehensive elastic properties from ELATE

    Examples
    --------
    >>> import numpy as np
    >>> C = np.eye(6) * 100  # Simple isotropic-like tensor
    >>> props = compute_elate_properties(C, density_g_cm3=5.0)
    >>> print(f"Hill bulk modulus: {props['hill_bulk_modulus_GPa']:.1f} GPa")
    """
    # Convert density: g/cm³ → kg/m³
    density_kg_m3 = density_g_cm3 * 1000.0

    analyzer = ElasticAnisotropyAnalyzer(C_voigt, density_kg_m3)
    return analyzer.compute_comprehensive_properties()
