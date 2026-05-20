# Synthesis Corpus

This directory contains a standalone experiment corpus for checking what the
stochastic synthesizer can and cannot currently recover. It is intentionally
separate from the unit test suite: results depend on random search, timeouts,
sample counts, and local machine speed.

## Layout

- `cases.jsonl`: input corpus. Each line is one JSON object with a trusted Miasm
  expression `repr` string and metadata.
- `reports/`: generated reports from local runs. This directory is ignored
  except for its `.gitignore`.

## Running

From the repository root:

```bash
python3 scripts/run_synthesis_corpus.py data/synthesis_corpus/cases.jsonl
python3 scripts/run_synthesis_corpus.py data/synthesis_corpus/cases.jsonl --tag constant
python3 scripts/run_synthesis_corpus.py data/synthesis_corpus/cases.jsonl --jobs 8
python3 scripts/run_synthesis_corpus.py data/synthesis_corpus/cases.jsonl --compare path/to/previous-report.json
```

The runner writes a JSON report and prints a short summary. It runs case/seed
attempts in parallel by default, using the host CPU count unless `--jobs` is
specified. A case is considered `solved` when at least one configured seed
produces a zero synthesis score and the synthesized expression is verified
equivalent to the original expression.

The starter corpus covers `x op y` and `x op large_const` cases for `+`, `-`,
`*`, `&`, `|`, `^`, `<<`, and `>>` across 8-, 32-, and 64-bit widths. It also
keeps a few repeated-variable, MBA identity, and `13337` constant markers.

## Case Format

Required fields:

- `id`: stable unique identifier.
- `expr`: trusted Miasm expression `repr` string.

Optional fields:

- `tags`: broad grouping labels for filtering.
- `features`: capability labels such as `variables`, `constants`, or `mul`.
- `samples`: synthesis oracle sample count for this case.
- `timeout`: per-seed local-search timeout in seconds.
- `seeds`: random seeds to try for this case.

Constants are represented as ordinary `ExprInt` literals in the expression.
The current stochastic grammar does not sample arbitrary literal terminals, but
some constants can still appear indirectly through expression identities and
Miasm simplification. Cases tagged with `constant` are useful markers for
tracking which constants are already reachable and which still need dedicated
constant synthesis support.
