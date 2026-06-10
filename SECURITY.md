# Security Policy

## Scope

This repository is Phase 1 research software. It does not execute trades or
hold exchange credentials. Security reports are still welcome for:

- secret or credential exposure
- unsafe deserialization of model artifacts
- path traversal or archive extraction issues
- dependency or supply-chain risks
- data-integrity failures
- temporal leakage that invalidates reported evidence
- artifact-hash or frozen-candidate bypasses

## Reporting

Do not publish sensitive details in a public issue. Use GitHub's private
vulnerability reporting for this repository when available, or contact the
repository owner privately through the GitHub profile.

Include:

- affected commit and file
- reproduction steps
- expected and observed behavior
- potential impact
- a minimal proof of concept when safe

## Secrets

Never commit:

- API keys or `.env` files
- exchange credentials
- Google Drive tokens
- raw account or state JSON
- model checkpoints containing private data

The current Phase 1 data pipeline uses public market data and does not require
authenticated Binance trading endpoints.

## Supported Version

Only the current `main` branch is actively maintained.
