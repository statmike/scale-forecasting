# NOTES.md — build log

Running log of deviations, surprises, and decisions made during the build. Append-only,
newest at the bottom. Keep entries short: what, why, and the contract section touched.

---

- **0.1 scaffold** — created the package tree per CONTRACTS §6 / DESIGN §6. All
  later-owned files are stubs raising `NotImplementedError` with a pointer to their BUILD
  step; `errors.py` is fully implemented (it's foundational and tiny). No deviations.
