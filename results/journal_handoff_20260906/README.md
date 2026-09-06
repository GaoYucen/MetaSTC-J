# MetaSTC-J journal handoff results — 2026-09-06

This is the auditable frozen result package for student continuation.

- `unified_results.md/csv/json`: authoritative paper-facing LSTM + FiLM five-seed comparison.
- `lstm_matched_controls/`: complete frozen LSTM V14 matched-control evidence (~2.9 MB), including per-seed metrics/checkpoints and diagnostic output.
- `film_five_seed/`: frozen FiLM V14 seeds 42–46, retaining per-run metrics, adapter checkpoint, run log and exit code.
- `diagnostics/film_target_protocol_audit.json`: paper/protocol provenance audit.
- `diagnostics/film_beijing_tuning_diagnostic.json`: bounded validation-only Beijing tuning diagnostic; it did not justify further tuning.

Treat this package as frozen evidence. Do not rerun or retune merely to obtain a more favorable number without a new PI decision.
