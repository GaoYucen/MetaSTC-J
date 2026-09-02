# MetaSTC-J Decision Log

## 2026-09-02 — Target dynamic heterogeneity
Move the journal problem from relatively static task heterogeneity toward dynamic spatio-temporal heterogeneity; this provides a clearer journal-level new problem than incremental static-clustering changes.

## 2026-09-02 — Use dynamic soft task discovery as the first core mechanism
Infer context-dependent latent task structure softly/dynamically and use it to condition downstream adaptation.

## 2026-09-02 — Selective adaptation is supporting, not the whole contribution
Test-time/selective fine-tuning should support the dynamic formulation, especially under shift, rather than stand alone as the main novelty.

## 2026-09-02 — Persist state outside local Codex chats
Use `AGENTS.md` and the four state files under `docs/` as the handoff layer across computers, ChatGPT, and Codex sessions.
