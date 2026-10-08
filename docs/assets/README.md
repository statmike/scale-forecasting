# `docs/assets/` — Static Documentation & Social Preview Assets

This directory holds static visual assets used by the MkDocs documentation site, the PyPI package description, and GitHub link previews.

```mermaid
flowchart LR
    HTML["social_card.html<br/>(HTML + Mermaid source)"] -->|"make social-card<br/>(headless Chrome 1280x640)"| PNG["social_card.png<br/>(1280x640 PNG)"]
    PNG --> OG["MkDocs / GitHub<br/>social preview card"]
    PNG --> PYPI["PyPI long description<br/>(static fallback)"]
```

## Directory Map

| File | Responsibility |
| :--- | :--- |
| [`README.md`](./README.md) | Directory map and regeneration instructions for static visual assets. |
| [`social_card.html`](./social_card.html) | Self-contained 1280×640 HTML + Mermaid source for the repository social preview card using the platform's four-class palette (`data`, `config`, `route`, `compute`). |
| [`social_card.png`](./social_card.png) | Rendered 1280×640 PNG used for GitHub repository social previews and static OpenGraph metadata. Regenerate via `make social-card`. |
