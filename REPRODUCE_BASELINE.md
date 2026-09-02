# Reproduction baseline: 2026-09-02

This branch freezes the 4090 reproduction state used immediately before journal-version technical modifications.

Canonical snapshot metadata, environment, hashes, result summaries, and lightweight logs are under `reproduction_snapshot/2026-09-02/`.

Large model checkpoints and datasets are intentionally not committed; their SHA256 hashes are recorded in the snapshot manifest/hash files. The server-side originals remain under `/workspace/MetaSTC-J/param/4090_tuned/` and `/workspace/MetaSTC-J/data/`.

Future journal experiments must not overwrite this branch or the server baseline outputs.
