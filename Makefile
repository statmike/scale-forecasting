# Developer entry points. The whole stack is aligned on uv: uv.lock + .python-version are the single
# source of truth for the interpreter and every dependency version, from local dev through CI, the
# runtime container, the Dataproc-cluster packed-venv, and the Colab notebooks. docker/requirements.txt
# is a DERIVED, human-readable export of the lock (kept in git so the container build and the
# content-addressed archive hash have a stable file to read) — regenerate it with `make lock`, never
# by hand.

# The exact export that produces docker/requirements.txt. Must stay identical here, in the file's
# header comment, and in the CI drift check (.github/workflows/ci.yml) — that three-way match is what
# guarantees the committed file always equals the lock.
EXPORT_ARGS := --frozen --no-emit-project --no-dev --no-hashes --extra models --extra ray --format requirements-txt

.PHONY: lock lock-check sync test docs composer-sync

## lock: re-resolve uv.lock from pyproject.toml and regenerate docker/requirements.txt from it.
## Run this after editing dependencies in pyproject.toml, then commit both files.
lock:
	uv lock
	uv export $(EXPORT_ARGS) -o docker/requirements.txt
	@echo "uv.lock + docker/requirements.txt regenerated — commit both."

## lock-check: fail if the lock is stale vs pyproject, or if requirements.txt drifted from the lock.
## This is the drift guard CI runs; run it locally before committing a dependency change.
lock-check:
	uv lock --check
	uv export $(EXPORT_ARGS) -o docker/requirements.txt.check
	@diff -u docker/requirements.txt docker/requirements.txt.check \
		&& rm -f docker/requirements.txt.check \
		&& echo "docker/requirements.txt is in sync with uv.lock" \
		|| { rm -f docker/requirements.txt.check; \
		     echo "ERROR: docker/requirements.txt is stale — run 'make lock' and commit."; exit 1; }

## sync: install the full dev environment (all extras + dev group) from the lock.
sync:
	uv sync --frozen --all-extras

## test: the offline test gate (no GCP / Spark / Ray required).
test:
	uv run ruff check src tests
	uv run pytest -m "not gcp and not spark and not ray" -q

## docs: build the documentation site (strict — fails on any broken nav/link/xref).
docs:
	uv run --group docs mkdocs build --strict

## composer-sync: deliver THIS working tree's src/ to the Composer workers (the code-delivery step).
## GitHub is only the origin — you pull/fork/modify locally, and this ships your src/ to the env so a
## worker imports the driver AND re-zips that same src/ to the jobs (custom models flow through, no
## image rebuild). Nothing at runtime pulls from GitHub. Run after `terraform apply
## -var create_composer=true`. Mirrors the package subtree (safe: won't touch unrelated plugins).
composer-sync:
	@prefix=$$(cd terraform/main && terraform output -raw composer_plugins_prefix 2>/dev/null); \
	 test -n "$$prefix" || { echo "ERROR: Composer not enabled — apply with create_composer=true first."; exit 1; }; \
	 echo "Delivering src/ -> $$prefix"; \
	 gsutil -m rsync -r -d src/scale_forecasting "$$prefix/scale_forecasting"; \
	 gsutil -m cp src/spark_main.py "$$prefix/spark_main.py"; \
	 echo "src/ delivered. Workers can now import the driver and ship this code to jobs."
