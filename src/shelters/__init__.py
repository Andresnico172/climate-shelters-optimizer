"""Climate-shelter location optimisation for Barcelona."""

from .instance import Instance, VulnerabilityWeights
from .objectives import MODELS, Objective, dmin_of, kpis

__all__ = ["Instance", "VulnerabilityWeights", "Objective", "MODELS", "dmin_of", "kpis"]
__version__ = "1.0.0"
