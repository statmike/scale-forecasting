"""Vertex AI Seq2Seq+ (`vertex_seq2seq`; ``automl`` family, ``vertex_automl`` runtime).

Managed Google Cloud sequence-to-sequence encoder-decoder forecasting architecture tuned for
medium-to-small panels (< 1 MB–100 MB) and fast experimentation with reduced search spaces.
Supports static attributes, historical past covariates, and known future covariates.

Official documentation:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/forecasting-train#seq2seq
"""

from __future__ import annotations

from ._vertex_automl_base import VertexAutoMLBaseModel
from .base_model import register


class VertexSeq2SeqModel(VertexAutoMLBaseModel):
    """Vertex AI Seq2Seq+ global forecasting model."""

    name = "vertex_seq2seq"
    package_url = (
        "https://docs.cloud.google.com/gemini-enterprise-agent-platform/"
        "machine-learning/tabular-data/tabular-workflows/forecasting-train#seq2seq"
    )
    supports_hierarchy_group_loss = False
    supports_custom_quantiles_in_workflow = False
    vertex_architecture_key = "seq2seq"


register(VertexSeq2SeqModel)
