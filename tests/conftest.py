"""Shared test configuration.

Markers are declared in ``pyproject.toml``; ``@gcp``/``@spark``/``@ray``/``@gpu`` tests
are collected but skipped unless their environment is available (wired per phase in Arc B).
"""

from __future__ import annotations
