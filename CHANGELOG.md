# Changelog

## [Unreleased]

### Fixed
- Refreshed security-sensitive dependency wiring by pinning `python-dotenv==1.2.2` and moving the `vllm` extra to the patched `0.20.0` line only for the supported Torch `2.11+` path. Older Torch branches now omit the `vllm` extra instead of resolving known-vulnerable vLLM releases.
