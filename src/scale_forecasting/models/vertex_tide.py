"""Vertex AI TiDE (`vertex_tide`; ``automl`` family, ``vertex_automl`` runtime).

Managed Google Cloud implementation of the Time-series Dense Encoder (Das et al., 2023;
https://arxiv.org/abs/2304.08424): an all-MLP encoder-decoder architecture with residual
connections that achieves up to 10x faster training than recurrent/attention models on large
multi-series panels while supporting static attributes, past covariates, future covariates, custom
quantile loss, and hierarchical group loss weighting (`hierarchy_group_columns`).

Official documentation:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting-train#time-series-dense-encoder
"""

from __future__ import annotations

from ._vertex_automl_base import VertexAutoMLBaseModel
from .base_model import register


class VertexTiDEModel(VertexAutoMLBaseModel):
    """Vertex AI Time-series Dense Encoder (TiDE) global forecasting model."""

    name = "vertex_tide"
    package_url = (
        "https://docs.cloud.google.com/gemini-enterprise-agent-platform/"
        "machine-learning/tabular-data/tabular-workflows/forecasting-train#time-series-dense-encoder"
    )
    supports_hierarchy_group_loss = True
    supports_custom_quantiles_in_workflow = True
    vertex_architecture_key = "tide"


register(VertexTiDEModel)
