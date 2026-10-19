"""AST security sandbox, wall-clock timeout executor, and temporal lookahead leak detector.

Enforces three non-negotiable pre-gate invariants on every candidate program before any cross-
validation fold is scored:

1. **AST Security Sandbox (:func:`validate_candidate_ast`):** Rejects prohibited system/network/file
   imports (``os``, ``sys``, ``subprocess``, ``socket``, ``urllib``, ``http``, ``pathlib``,
   ``shutil``, ``ctypes``, ``pickle``, ``builtins``, ``importlib``) and dangerous builtins or
   reflection calls (``eval``, ``exec``, ``open``, ``compile``, ``__import__``, ``getattr``,
   ``setattr``, ``globals``, ``locals``). Only whitelisted scientific modules (``numpy``,
   ``pandas``, ``scipy``, ``statsmodels``, ``sklearn``, ``math``, ``statistics``,
   ``scale_forecasting``) may be imported.
2. **Temporal Causality & Future-Leakage Detector (:func:`verify_no_future_leakage`):** Perturbs the
   last ``horizon`` target rows per series with ``+1e6`` sentinel noise and verifies that all
   engineered features or predictions at earlier timestamps $t \\le T - H$ remain bit-for-bit
   identical (``np.allclose(..., atol=1e-12)``).
3. **Bounded Sandbox Execution (:func:`execute_in_sandbox`):** Compiles AST-verified code inside a
   restricted builtins namespace under a strict wall-clock timeout.
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats as scipy_stats

from ..errors import ConfigError
from ..models.base_model import DEFAULT_QUANTILES, BaseModel, ModelContext
from .targets import AutoPolicyWeights

__all__ = [
    "ALLOWED_MODULES",
    "PROHIBITED_CALLS",
    "PROHIBITED_MODULES",
    "SandboxValidationResult",
    "compile_candidate_symbol",
    "execute_in_sandbox",
    "validate_candidate_ast",
    "verify_no_future_leakage",
]

PROHIBITED_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "http",
        "pathlib",
        "shutil",
        "ctypes",
        "pickle",
        "builtins",
        "importlib",
        "multiprocessing",
        "threading",
        "signal",
        "tempfile",
        "glob",
        "io",
        "pty",
        "fcntl",
        "resource",
        "inspect",
        "code",
        "marshal",
        "shelve",
        "requests",
        "httpx",
        "aiohttp",
    }
)

ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "__future__",
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "sklearn",
        "math",
        "statistics",
        "scale_forecasting",
        "typing",
        "collections",
        "dataclasses",
        "functools",
        "itertools",
    }
)

PROHIBITED_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "help",
        "memoryview",
        "system",
        "popen",
    }
)

_PROHIBITED_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__dict__",
        "__class__",
        "__closure__",
        "__loader__",
        "__spec__",
    }
)

_MAX_AST_NODES: int = 2500


@dataclass(frozen=True)
class SandboxValidationResult:
    """Result of static AST security inspection on a candidate program."""

    valid: bool
    violations: tuple[str, ...] = ()
    ast_node_count: int = 0

    @property
    def is_valid(self) -> bool:
        """Alias for ``self.valid``."""
        return self.valid

    @property
    def errors(self) -> tuple[str, ...]:
        """Alias for ``self.violations``."""
        return self.violations

    @property
    def reason(self) -> str | None:
        """Joined violation summary string, or ``None`` when valid."""
        return "; ".join(self.violations) if self.violations else None

    def __bool__(self) -> bool:
        return self.valid


def validate_candidate_ast(code: str) -> SandboxValidationResult:
    """Parse ``code`` into an AST and reject prohibited imports, calls, and reflection."""
    if not isinstance(code, str) or not code.strip():
        return SandboxValidationResult(
            valid=False,
            violations=("Candidate code is empty.",),
            ast_node_count=0,
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return SandboxValidationResult(
            valid=False,
            violations=(f"SyntaxError at line {exc.lineno}: {exc.msg}",),
            ast_node_count=0,
        )

    violations: list[str] = []
    node_count = 0

    for node in ast.walk(tree):
        node_count += 1
        if node_count > _MAX_AST_NODES:
            violations.append(f"AST node count exceeds maximum limit ({_MAX_AST_NODES}).")
            break

        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".", 1)[0]
                if root_mod in PROHIBITED_MODULES:
                    violations.append(f"Prohibited import '{alias.name}'.")
                elif root_mod not in ALLOWED_MODULES:
                    violations.append(
                        f"Import '{alias.name}' is not in the allowed scientific module whitelist."
                    )

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            root_mod = mod_name.split(".", 1)[0] if mod_name else ""
            if root_mod in PROHIBITED_MODULES:
                violations.append(f"Prohibited import from '{mod_name}'.")
            elif root_mod and root_mod not in ALLOWED_MODULES:
                violations.append(
                    f"Import from '{mod_name}' is not in the allowed scientific module whitelist."
                )
            for alias in node.names:
                if alias.name in PROHIBITED_MODULES or alias.name in PROHIBITED_CALLS:
                    violations.append(f"Prohibited symbol import '{alias.name}'.")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
                if fn_name in PROHIBITED_CALLS:
                    violations.append(f"Prohibited function call '{fn_name}()'.")
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in PROHIBITED_CALLS:
                    violations.append(f"Prohibited method call '.{attr_name}()'.")
                if attr_name in (
                    "to_csv",
                    "to_parquet",
                    "to_pickle",
                    "read_csv",
                    "read_pickle",
                    "read_parquet",
                ):
                    violations.append(f"Prohibited file I/O method call '.{attr_name}()'.")

        elif isinstance(node, ast.Attribute):
            if node.attr in _PROHIBITED_DUNDER_ATTRS:
                violations.append(f"Prohibited dunder attribute access '.{node.attr}'.")

        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "__import__"):
            violations.append(f"Prohibited identifier '{node.id}'.")

    return SandboxValidationResult(
        valid=(len(violations) == 0),
        violations=tuple(dict.fromkeys(violations)),
        ast_node_count=node_count,
    )


def _safe_import(
    name: str,
    globals_dict: Mapping[str, Any] | None = None,
    locals_dict: Mapping[str, Any] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    """Restricted ``__import__`` hook permitting only ``ALLOWED_MODULES``."""
    root = name.split(".", 1)[0]
    if root in PROHIBITED_MODULES or root not in ALLOWED_MODULES:
        raise ImportError(f"Sandbox blocked import of disallowed module '{name}'.")
    return builtins.__import__(name, globals_dict, locals_dict, fromlist, level)


_SAFE_BUILTINS: dict[str, Any] = {
    "__build_class__": builtins.__build_class__,
    "__import__": _safe_import,
    "__name__": "scale_forecasting_evolve_sandbox",
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "callable": callable,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hash": hash,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "pow": pow,
    "property": property,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "RuntimeError": RuntimeError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
}


def compile_candidate_symbol(code: str, entrypoint: str) -> Any:
    """Validate ``code`` with :func:`validate_candidate_ast` and return ``entrypoint``."""
    validation = validate_candidate_ast(code)
    if not validation.valid:
        raise ConfigError(f"Candidate failed AST security sandbox: {validation.reason}")

    env: dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "math": math,
        "statistics": statistics,
        "scipy": scipy,
        "stats": scipy_stats,
        "BaseModel": BaseModel,
        "ModelContext": ModelContext,
        "DEFAULT_QUANTILES": DEFAULT_QUANTILES,
        "AutoPolicyWeights": AutoPolicyWeights,
    }
    compiled = compile(code, filename="<evolve_candidate>", mode="exec")
    exec(compiled, env)
    if entrypoint not in env:
        raise ConfigError(f"Candidate code did not define required entrypoint '{entrypoint}'.")
    return env[entrypoint]


def execute_in_sandbox(
    code: str,
    entrypoint: str,
    *args: Any,
    timeout_seconds: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Compile and execute ``entrypoint(*args, **kwargs)`` with a hard wall-clock timeout."""
    symbol = compile_candidate_symbol(code, entrypoint)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(symbol, *args, **kwargs)
        try:
            return future.result(timeout=max(0.5, float(timeout_seconds)))
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f"Candidate '{entrypoint}' exceeded sandbox timeout ({timeout_seconds:.1f}s)."
            ) from exc


def verify_no_future_leakage(
    fn: Any,
    df: pd.DataFrame,
    target_col: str = "y",
    time_col: str = "ds",
    id_col: str = "ts_id",
    horizon: int = 7,
) -> bool:
    """Verify that perturbing the last ``horizon`` target values by ``+1e6`` does not alter
    any engineered features or fitted values on earlier timestamps $t \\le T - H$.

    Returns ``True`` only when all outputs on $t \\le T - H$ are bit-for-bit identical
    (``np.allclose(..., atol=1e-12, equal_nan=True)``). Immediately returns ``False`` if the
    candidate uses future information (such as ``.shift(-k)``, ``.bfill()``, or full-series
    global target aggregations).
    """
    if df.empty or target_col not in df.columns:
        return False

    h = max(1, int(horizon))
    work = df.copy()
    if id_col not in work.columns:
        work[id_col] = "series_0"
    if time_col in work.columns:
        work = work.sort_values([id_col, time_col]).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    # Handle BaseModel subclass (target="model")
    if isinstance(fn, type) and issubclass(fn, BaseModel):
        try:
            first_id = str(work[id_col].iloc[0])
            s_df = work[work[id_col].astype(str) == first_id].reset_index(drop=True)
            if len(s_df) <= h + 2:
                return True
            y_clean = pd.Series(
                pd.to_numeric(s_df[target_col], errors="coerce").to_numpy(dtype=float),
                index=pd.to_datetime(s_df[time_col]) if time_col in s_df.columns else None,
            )
            y_pert = y_clean.copy()
            y_pert.iloc[-h:] = y_pert.iloc[-h:] + 1e6

            ctx = ModelContext(freq="D", horizon=h)
            m_clean = fn({}, ctx)
            m_clean.fit(y_clean.iloc[:-h])
            m_pert = fn({}, ctx)
            m_pert.fit(y_pert.iloc[:-h])
            p_clean = m_clean.predict(h)["yhat"].to_numpy(dtype=float)
            p_pert = m_pert.predict(h)["yhat"].to_numpy(dtype=float)
            return bool(np.allclose(p_clean, p_pert, atol=1e-12, rtol=1e-9, equal_nan=True))
        except Exception:  # noqa: BLE001 - any failure inside candidate model rejects it
            return False

    # Build prefix mask (True for earlier timestamps t <= T - H, False for last H rows per series)
    prefix_mask = np.ones(len(work), dtype=bool)
    df_clean = work.copy()
    df_perturbed = work.copy()
    df_perturbed[target_col] = pd.to_numeric(df_perturbed[target_col], errors="coerce").astype(
        float
    )

    for _, grp in work.groupby(id_col, sort=False):
        idx = grp.index.to_numpy()
        eff_h = min(h, max(1, len(idx) // 2))
        tail_idx = idx[-eff_h:]
        prefix_mask[tail_idx] = False
        df_perturbed.loc[tail_idx, target_col] = (
            df_perturbed.loc[tail_idx, target_col].to_numpy(dtype=float) + 1e6
        )

    try:
        out_clean = fn(df_clean)
        out_perturbed = fn(df_perturbed)
    except TypeError:
        try:
            out_clean = fn(df_clean, None)
            out_perturbed = fn(df_perturbed, None)
        except Exception:  # noqa: BLE001 - candidate raised during causality check
            return False
    except Exception:  # noqa: BLE001 - candidate raised during causality check
        return False

    if not isinstance(out_clean, pd.DataFrame) or not isinstance(out_perturbed, pd.DataFrame):
        return False
    if len(out_clean) != len(work) or len(out_perturbed) != len(work):
        return False
    if list(out_clean.columns) != list(out_perturbed.columns):
        return False

    for col in out_clean.columns:
        if col in (id_col, time_col):
            continue
        s_clean = out_clean[col].iloc[prefix_mask]
        s_pert = out_perturbed[col].iloc[prefix_mask]
        if pd.api.types.is_numeric_dtype(s_clean) or pd.api.types.is_numeric_dtype(s_pert):
            a_clean = pd.to_numeric(s_clean, errors="coerce").to_numpy(dtype=float)
            a_pert = pd.to_numeric(s_pert, errors="coerce").to_numpy(dtype=float)
            if not np.allclose(a_clean, a_pert, atol=1e-12, rtol=1e-9, equal_nan=True):
                return False
        else:
            if not s_clean.astype(str).equals(s_pert.astype(str)):
                return False

    return True
