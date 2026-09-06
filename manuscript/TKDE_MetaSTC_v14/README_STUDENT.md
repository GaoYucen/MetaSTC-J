# TKDE MetaSTC journal manuscript handoff

Primary source: `TKDE_MetaSTC_Journal.tex`. `TKDE_MetaSTC_original_20260906.tex` is the pre-edit snapshot. `TKDE_MetaSTC_Journal_review.pdf` is a compact visual reference.

Build:
```bash
pdflatex TKDE_MetaSTC_Journal.tex
bibtex TKDE_MetaSTC_Journal
pdflatex TKDE_MetaSTC_Journal.tex
pdflatex TKDE_MetaSTC_Journal.tex
```
Build intermediates are intentionally not tracked.
