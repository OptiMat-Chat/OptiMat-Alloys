"""
Session State Management for Memory Layer

Tracks calculation parameters and detects changes between calculations
to enable the agent to recognize when parameters have changed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class CalculationParameters:
    """Parameters used for a calculation."""
    calculator: str
    supercell_size: int
    fmax: float = 0.001
    timestamp: datetime = field(default_factory=datetime.now)

    def differs_from(self, other: Optional["CalculationParameters"]) -> List[str]:
        """
        Return list of parameter names that differ from another CalculationParameters.

        Args:
            other: Another CalculationParameters to compare, or None

        Returns:
            List of parameter names that differ (e.g., ["calculator", "supercell_size"])
        """
        if other is None:
            return []

        changes = []
        if self.calculator != other.calculator:
            changes.append("calculator")
        if self.supercell_size != other.supercell_size:
            changes.append("supercell_size")
        if abs(self.fmax - other.fmax) > 1e-6:
            changes.append("fmax")
        return changes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calculator": self.calculator,
            "supercell_size": self.supercell_size,
            "fmax": self.fmax,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SessionState:
    """
    Tracks session state and parameter changes for the memory layer.

    This class maintains awareness of:
    - Current calculation parameters (from settings)
    - Parameters used in the last calculation
    - Changes between settings and last calculation

    The agent uses this information to provide context-aware responses,
    such as noting when a calculator has changed since the last structure
    was generated.
    """
    current_params: Optional[CalculationParameters] = None
    last_calculation_params: Optional[CalculationParameters] = None
    calculation_count: int = 0

    def update_current_params(
        self,
        calculator: str,
        supercell_size: int,
        fmax: float = 0.001
    ) -> None:
        """
        Update current parameters when settings change.

        Called when:
        - Session starts (from initial settings)
        - User changes settings via on_settings_update

        Args:
            calculator: Current calculator name
            supercell_size: Current default supercell size
            fmax: Current force convergence threshold
        """
        self.current_params = CalculationParameters(
            calculator=calculator,
            supercell_size=supercell_size,
            fmax=fmax,
            timestamp=datetime.now()
        )

    def record_calculation(
        self,
        calculator: Optional[str] = None,
        supercell_size: Optional[int] = None,
        fmax: float = 0.001
    ) -> None:
        """
        Record that a calculation was performed.

        Called by tools after completing a calculation. This snapshots
        the parameters used so we can detect changes before the next calculation.

        Args:
            calculator: Calculator used (defaults to current_params.calculator)
            supercell_size: Supercell size used (defaults to current_params.supercell_size)
            fmax: Force convergence used
        """
        # Use provided values or fall back to current params
        calc = calculator if calculator is not None else (
            self.current_params.calculator if self.current_params else "unknown"
        )
        size = supercell_size if supercell_size is not None else (
            self.current_params.supercell_size if self.current_params else 0
        )

        self.last_calculation_params = CalculationParameters(
            calculator=calc,
            supercell_size=size,
            fmax=fmax,
            timestamp=datetime.now()
        )
        self.calculation_count += 1

    def get_changes_since_last_calculation(self) -> List[str]:
        """
        Detect what parameters changed since the last calculation.

        Returns:
            List of parameter names that differ between current settings
            and the last calculation (e.g., ["calculator", "supercell_size"])
        """
        if self.current_params is None or self.last_calculation_params is None:
            return []
        return self.current_params.differs_from(self.last_calculation_params)

    def generate_context_block(self) -> str:
        """
        Generate context string for system message injection.

        Returns:
            Formatted context block describing current parameters and changes,
            or empty string if no context to add.
        """
        if self.current_params is None:
            return ""

        # Build current parameters section
        lines = [
            "## SESSION CONTEXT (for your awareness - only reference if relevant to user's request)",
            "",
            "Current parameters:",
            f"- Calculator: `{self.current_params.calculator}`",
            f"- Default Supercell Size: {self.current_params.supercell_size} atoms",
            f"- Force Convergence (fmax): {self.current_params.fmax} eV/A",
        ]

        # Add changes section if there are changes
        changes = self.get_changes_since_last_calculation()
        if changes and self.last_calculation_params is not None:
            lines.append("")
            lines.append("Changes since last calculation:")

            if "calculator" in changes:
                lines.append(
                    f"- Calculator: `{self.last_calculation_params.calculator}` -> "
                    f"`{self.current_params.calculator}`"
                )
            if "supercell_size" in changes:
                lines.append(
                    f"- Supercell Size: {self.last_calculation_params.supercell_size} -> "
                    f"{self.current_params.supercell_size} atoms"
                )
            if "fmax" in changes:
                lines.append(
                    f"- Force Convergence: {self.last_calculation_params.fmax} -> "
                    f"{self.current_params.fmax} eV/A"
                )

        # Add calculation count for context
        if self.calculation_count > 0:
            lines.append("")
            lines.append(f"Session calculations completed: {self.calculation_count}")

        # Add guidance note
        lines.append("")
        lines.append(
            "Note: Only mention these parameters if the user asks about calculations "
            "or if changes are relevant to their request."
        )

        return "\n".join(lines)

    def has_parameter_changes(self) -> bool:
        """Check if there are any parameter changes since the last calculation."""
        return len(self.get_changes_since_last_calculation()) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary for debugging/logging."""
        return {
            "current_params": self.current_params.to_dict() if self.current_params else None,
            "last_calculation_params": (
                self.last_calculation_params.to_dict()
                if self.last_calculation_params else None
            ),
            "calculation_count": self.calculation_count,
            "has_changes": self.has_parameter_changes(),
            "changes": self.get_changes_since_last_calculation()
        }
