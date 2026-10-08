# Security Policy

## Supported versions

| Version | Supported |
| :--- | :--- |
| `1.0.x` | Yes |
| `< 1.0.0` | No |

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, report vulnerabilities privately through **[GitHub Security Advisories](https://github.com/statmike/scale-forecasting/security/advisories/new)** on this repository. Include:

1. A description of the issue and its potential impact.
2. The affected component (`src/scale_forecasting/...`, `terraform/...`, `docker/...`, or `.github/workflows/...`).
3. Minimal steps or a configuration snippet to reproduce the issue.

We will acknowledge receipt within 5 business days and coordinate a fix and disclosure timeline with you.

---

## Security architecture & credential policy

`scale-forecasting` is designed to run on Google Cloud with zero static secrets:

1. **No service account keys:** Neither local development nor cloud execution uses downloaded JSON key files (`*.json.key`). Local CLI and SDK workflows authenticate via short-lived Application Default Credentials (`gcloud auth application-default login`), and cloud workloads run under attached, least-privilege IAM service accounts provisioned by [`terraform/main/modules/iam`](./terraform/main/modules/iam/main.tf).
2. **Enforced Public Access Prevention:** Every Cloud Storage bucket provisioned by [`terraform/bootstrap`](./terraform/bootstrap/main.tf) and [`terraform/main/modules/storage`](./terraform/main/modules/storage/main.tf) sets `public_access_prevention = "enforced"` and `uniform_bucket_level_access = true`.
3. **Private networking:** Cloud runtimes execute on a dedicated VPC network with Private Google Access enabled ([`terraform/main/modules/network`](./terraform/main/modules/network/main.tf)).
4. **Provenance attestations on releases:** Official wheels and source distributions on PyPI are published exclusively from GitHub Actions ([`.github/workflows/release.yml`](./.github/workflows/release.yml)) via OpenID Connect (OIDC) Trusted Publishing with PEP 740 cryptographic attestations.
