"""Vertex AI Temporal Fusion Transformer (`vertex_tft`; ``automl`` family, ``vertex_automl``).

Managed Google Cloud implementation of the Temporal Fusion Transformer (Lim et al., 2021;
https://arxiv.org/abs/1912.09363): combines variable selection networks, static covariate encoders,
gated residual networks, and interpretable multi-head attention across historical lookback and
future horizons. Emits built-in attention-based encoder/decoder/static feature importances alongside
Shapley/Integrated Gradients attributions.

Official documentation:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting-train#temporal-fusion-transformer
"""

from __future__ import annotations

from ._vertex_automl_base import VertexAutoMLBaseModel
from .base_model import register


class VertexTFTModel(VertexAutoMLBaseModel):
    """Vertex AI Temporal Fusion Transformer (TFT) global forecasting model."""

    name = "vertex_tft"
    package_url = (
        "https://docs.cloud.google.com/gemini-enterprise-agent-platform/"
        "machine-learning/tabular-data/tabular-workflows/forecasting-train#temporal-fusion-transformer"
    )
    supports_hierarchy_group_loss = False
    supports_custom_quantiles_in_workflow = False
    vertex_architecture_key = "tft"


register(VertexTFTModel)
