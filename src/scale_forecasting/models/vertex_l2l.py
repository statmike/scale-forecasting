"""Vertex AI Learn-to-Learn (`vertex_l2l`; ``automl`` family, ``vertex_automl`` runtime).

Google Cloud's flagship AutoML Forecasting architecture: conducts Stage-1 Neural Architecture Search
(NAS) across candidate encoder-decoder neural networks followed by Stage-2 bagged ensembling over
the top selected trials. Supports static attributes, historical past covariates, known future
covariates, custom quantile loss, and hierarchical group loss weighting (`hierarchy_group_columns`).

Official documentation:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting-train#l2l
"""

from __future__ import annotations

from ._vertex_automl_base import VertexAutoMLBaseModel
from .base_model import register


class VertexL2LModel(VertexAutoMLBaseModel):
    """Vertex AI AutoML / Learn-to-Learn (L2L) global forecasting model."""

    name = "vertex_l2l"
    package_url = (
        "https://docs.cloud.google.com/gemini-enterprise-agent-platform/"
        "machine-learning/tabular-data/tabular-workflows/forecasting-train#l2l"
    )
    supports_hierarchy_group_loss = True
    supports_custom_quantiles_in_workflow = True
    vertex_architecture_key = "l2l"


register(VertexL2LModel)
