# msynth Datasets

This directory contains versioned corpus data used by msynth experiments and
scripts. Corpus files live under [`corpora/`](corpora/).

These datasets provide stable inputs for evaluating MBA handling in msynth. The
CoBRA-derived corpus is meant to stress parsing and oracle-backed simplification
on a broad set of real MBA benchmark expressions; the real-world corpus is a large
set of MBAs harvested from production obfuscated software; and the synthesis corpus
is a smaller controlled suite for checking stochastic synthesis behavior.

## Layout

- `corpora/cobra.jsonl.gz`: compact source-text corpus generated from Trail of
  Bits CoBRA test datasets.
- `corpora/realworld.jsonl.gz`: large corpus of real-world MBAs mined from
  production obfuscated software.
- `corpora/synthesis.jsonl.gz`: small hand-written stochastic synthesis corpus
  for checking synthesizer behavior.
- `corpora/cegis.jsonl.gz`: constant-bearing MBA corpus for the CEGIS
  constant-synthesis path (see below).

## CEGIS Constant-Synthesis Corpus

Path:

```text
datasets/corpora/cegis.jsonl.gz
```

A controlled suite of **obfuscated, equivalent-but-larger** forms of
constant-bearing expressions, each paired with its minimal constant-bearing
`expected_miasm` (IR encoding, like the real-world corpus). Each obfuscated input
is built from its minimal form via exact MBA identities (`a+b = (a^b)+2(a&b)`,
`a^b = (a|b)-(a&b)`, `a|b = (a^b)+(a&b)`, and expanded products), so the recorded
ground truth is a true equivalent of a strictly larger input. Evaluated by the
standard runner:

```text
python3 scripts/run_simplification_corpus.py \
    --corpus datasets/corpora/cegis.jsonl.gz --empty-oracle --limit 0 \
    --success-metric covered [--cegis]
```

Because every input is obfuscated, the `covered` metric (output equivalent to the
input *and* no larger than the expected node count) measures whether the pipeline
recovers the minimal form. The `--cegis` flag isolates the constant solver's
contribution:

- linear categories (`c*x+k`, `c0*x+c1*y+k`, `c*(x+y)`, `(x&c0)|(y&c1)`) are
  reached by SiMBA/GAMBA alone — the no-CEGIS baseline;
- non-linear categories (expanded products `(x+c0)*(y+c1)`, quadratics, multiply-
  xor `(c0*x)^(c1*y)`) are structurally outside the linear simplifiers and are
  recovered only with `--cegis`.

On the shipped suite this is **covered 75/120 without CEGIS → 120/120 with CEGIS**
(`not_equivalent=0`)..

## Real-World Corpus

Path:

```text
datasets/corpora/realworld.jsonl.gz
```

Format:

```json
{
  "id": "case_000000",
  "source": "game",
  "expr_miasm": "ExprOp('-', ExprOp('<<', ExprId('x0', size=32), ExprInt(0x2, 32)), ExprInt(0x1, 32))",
  "size": 32
}
```

Unlike the CoBRA corpus (infix `expr_text`), this corpus stores each expression as a
**Miasm IR repr** in `expr_miasm`, read back via `eval`
(`msynth.parsing.parse_expr`). The IR encoding is required because real-world MBAs
use slices, `ExprCompose`, zero/sign-extension and multi-size memory that the infix
grammar cannot represent. There is no ground-truth (`expected_*`) field: these
are obfuscated expressions with no known canonical simplification, so the corpus
runner scores them by node-count reduction only.

### Provenance

Real-world MBA expressions mined from production obfuscated / protected software
(commercial copy protection, VM-based obfuscators, and anti-cheat engines) via
symbolic execution over their virtualized handlers. Each expression is normalized:
every terminal (registers and memory accesses, with memory treated as one opaque
variable) is unified to `x0, x1, …` in first-appearance order and no simplification is applied 
to preserve the obfuscated shape. Structurally identical normalized expressions are deduplicated.
The `source` field is generalized to one of:

- `game` — expressions from protected games,
- `anticheat` — expressions from anti-cheat engines,
- `other` — expressions from other obfuscated binaries.

The corpus currently contains 362,991 rows (`game` 361,145, `anticheat` 1,175,
`other` 671).

Read it with `gzip.open(..., "rt")`. Evaluate it with the same runner as the CoBRA
corpus — it auto-detects the IR encoding:

```bash
python3 scripts/run_simplification_corpus.py \
  --corpus datasets/corpora/realworld.jsonl.gz \
  --oracle oracle.pickle \
  --mode AST \
  --limit 0 \
  [--sources game]
```

## CoBRA Corpus

Path:

```text
datasets/corpora/cobra.jsonl.gz
```

Format:

```json
{
  "id": "case_000000",
  "source": "gamba/loki_tiny.txt",
  "suite": "gamba",
  "expr_text": "(~ ((~ x) + (- (x ^ (x ^ y)))))",
  "expected_text": "x + y",
  "size": 64
}
```

The file is gzip-compressed JSONL. Read it with `gzip.open(..., "rt")`; do not
check in an uncompressed copy. The expression strings are parsed into Miasm IR
at load time with `msynth.parsing.parse_infix_expr`.

The generated corpus currently contains 76,559 rows from all `.txt` files under
CoBRA `test/datasets`.

Suite counts:

- `gamba`: 41,000
- `simba`: 31,581
- `msimba`: 1,000
- `multivariate64`: 1,000
- `univariate64`: 1,000
- `oses`: 958
- `permutation64`: 13
- `obfuscatorx`: 7

Original sources referenced by the
[upstream CoBRA dataset](https://github.com/trailofbits/CoBRA/tree/master/test/datasets)
include:

- Trail of Bits CoBRA: https://github.com/trailofbits/CoBRA
- SiMBA datasets: https://github.com/DenuvoSoftwareSolutions/SiMBA and
  https://github.com/pgarba/SiMBA-
- GAMBA datasets: https://github.com/DenuvoSoftwareSolutions/GAMBA
- MBA-Blast: https://github.com/softsec-unh/MBA-Blast
- MBA-Solver: https://github.com/softsec-unh/MBA-Solver
- NeuReduce: https://github.com/fvrmatteo/NeuReduce
- OSES datasets:
  https://github.com/fvrmatteo/oracle-synthesis-meets-equality-saturation
- Mazeworks Simplifier: https://github.com/mazeworks-security/Simplifier

Some upstream headers note missing or unknown licenses for individual source
datasets. Keep those provenance notes in mind before redistributing derived
artifacts outside this repository.

Regenerate from a local CoBRA checkout:

```bash
python3 scripts/preprocess_dataset.py ../CoBRA/test/datasets \
  --size 64 \
  --passes none \
  --missing-expected self \
  --fail-fast \
  --output datasets/corpora/cobra.jsonl.gz
```

`--passes none` is intentional for this canonical corpus: the file stores
source expressions, while parsing to Miasm IR is validated during generation.
Rows whose upstream ground-truth field is `-` are stored with
`expected_text == expr_text` via `--missing-expected self`.

Check oracle-backed simplification against the first N corpus rows:

```bash
python3 scripts/run_simplification_corpus.py \
  --corpus datasets/corpora/cobra.jsonl.gz \
  --oracle oracle.pickle \
  --limit 1000
```

The script parses each selected `expr_text`, simplifies it with msynth, and
checks whether the result first reaches `expected_text`; if not, the result must
still have fewer Miasm graph nodes than the original expression. Simplifier
exceptions are reported as failing `error` rows; they are not filtered out of the
checked prefix. Parallel workers are initialized once and reuse their loaded
simplifier for all assigned rows. By default, the script uses all available CPU
cores, and `--jobs` can be used to override that. Failure reports include both a
readable Miasm expression string (`simplified`) and the exact Miasm `repr` form
(`simplified_ir`).

## Synthesis Corpus

Path:

```text
datasets/corpora/synthesis.jsonl.gz
```

Format:

```json
{
  "id": "var_x_u8",
  "expr": "ExprId('p0', 8)",
  "expected_expr": "ExprId('p0', 8)",
  "expr_text": "ExprId('p0', 8)",
  "expected_text": "ExprId('p0', 8)",
  "tags": ["identity", "variables", "u8"],
  "features": ["variables"]
}
```

This corpus is gzip-compressed JSONL and uses trusted Miasm `repr` strings
because it targets the stochastic synthesis runner directly. Runtime parameters
such as samples, timeout, seeds, and job count are configured by
`scripts/run_synthesis_corpus.py`, not stored in the corpus rows.

Run it with:

```bash
python3 scripts/run_synthesis_corpus.py --output reports/synthesis-corpus.json
python3 scripts/run_synthesis_corpus.py --tag constant --output reports/constants.json
python3 scripts/run_synthesis_corpus.py --jobs 8 --output reports/parallel.json
```
