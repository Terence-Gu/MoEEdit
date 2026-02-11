"""
MoE Knowledge Editing Package
"""

from .utils import EditRequest
from .manager import ProjectionMatrixManager
from .manager import Qwen3MoEStatisticsCollector
from .optim import BlockCoordinateDescent
from .config import MoEEditConfig
from .editor import MoEKnowledgeEditor

__version__ = "1.0.0"
__all__ = [
    "MoEKnowledgeEditor",
    "EditRequest",
    "ProjectionMatrixManager",
    "Qwen3MoEStatisticsCollector",
    "BlockCoordinateDescent",
    "pre_edit_prob",
    "MoEEditConfig"
]