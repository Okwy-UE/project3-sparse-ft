"""
Model utilities and wrappers for sparse fine-tuning.
"""

from .sparse_model_wrapper import SparseModelWrapper, prepare_sparse_model

__all__ = ['SparseModelWrapper', 'prepare_sparse_model']
