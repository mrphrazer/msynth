# msynth
Author: **Tim Blazytko** and **Moritz Schloegel**

msynth is a code deobfuscation framework to simplify Mixed Boolean-Arithmetic (MBA) expressions. Given a pre-computed simplification oracle, it walks over a complex expression represented as an abstract syntax tree (AST) and tries to simplify subtrees based on oracle lookups. Alternatively, it tries to simplify expressions via stochastic program synthesis.  

msynth is built on top of [Miasm](https://github.com/cea-sec/miasm) and inspired by the papers 

* ["QSynth: A Program Synthesis based Approach for Binary Code Deobfuscation"](https://archive.bar/pdfs/bar2020-preprint9.pdf) by Robin David, Luigi Coniglio and Mariano Ceccato (NDSS, BAR 2020),

* ["Syntia: Synthesizing the Semantics of Obfuscated Code"](https://synthesis.to/papers/usenix17-syntia.pdf) by Tim Blazytko, Moritz Contag, Cornelius Aschermann and Thorsten Holz (USENIX Security 2017) and

* ["Search-Based Local Blackbox Deobfuscation: Understand, Improve and Mitigate"](https://binsec.github.io/assets/publications/papers/2021-ccs.pdf) by Grégoire Menguy, Sébastien Bardin, Richard Bonichon and Cauim de Souza de Lima (CCS 2021).


It can be used in combination with Miasm's symbolic execution engine to simplify complex expressions in obfuscated code or as a standalone tool to play around with MBA simplification.

```
original: {((((((((RSI[0:32] ^ 0xFFFFFFFF) & RDX[0:32]) + RSI[0:32]) ^ 0xFFFFFFFF) & RDX[0:32]) + ((RSI[0:32] ^ 0xFFFFFFFF) & RDX[0:32]) + RSI[0:32]) & (RDX[0:32] ^ 0xFFFFFFFF)) + -(((((((RSI[0:32] ^ 0xFFFFFFFF) & RDX[0:32]) + RSI[0:32]) ^ 0xFFFFFFFF) & RDX[0:32]) + ((RSI[0:32] ^ 0xFFFFFFFF) & RDX[0:32]) + RSI[0:32]) | (((RSI[0:32] ^ 0xFFFFFFFF) & RDX[0:32]) + RSI[0:32])) + ({RDI[0:32] & ({RDI[0:32] & RSI[0:32] 0 32, 0x0 32 64} * 0x2 + {RDI[0:32] ^ RSI[0:32] 0 32, 0x0 32 64})[0:32] 0 32, 0x0 32 64} * 0x2 + {((((RSI[0:32] ^ 0xFFFFFFFF) & RDX[0:32]) + RSI[0:32]) & RSI[0:32]) + (((RDI + {(RDI[0:32] ^ 0xFFFFFFFF) | RDX[0:32] 0 32, 0x0 32 64} + 0x1)[0:32] ^ 0xFFFFFFFF) & RDX[0:32]) + (RDI[0:32] ^ ({RDI[0:32] & RSI[0:32] 0 32, 0x0 32 64} * 0x2 + {RDI[0:32] ^ RSI[0:32] 0 32, 0x0 32 64})[0:32]) + ((RDI[0:32] ^ 0xFFFFFFFF) | RDX[0:32]) + (RDI + RDX + 0x1)[0:32] 0 32, 0x0 32 64})[0:32]) * 0x2 0 32, 0x0 32 64}

simplified: {(-RDX[0:32] + ((RDI[0:32] + RDX[0:32] + RSI[0:32]) << 0x1)) * 0x2 0 32, 0x0 32 64}
```


## Core Features

* simplifies most MBAs found in the wild
* can simplify whole expressions to constants
* makes uses of large pre-computed lookup tables (for efficiency)
* can verify the soundness of simplifications with an SMT solver
* can learn expressions from input-output behavior
* supports parallelization
* fully integrable into Miasm's symbolic execution engine

## Installation
To install msynth follow these steps:

```
git clone https://github.com/mrphrazer/msynth.git
cd msynth

# optionally: use a virtual environment
python -m venv msynth-env
source msynth-env/bin/activate

# install dependencies
pip install -r requirements.txt

# install msynth
pip install .

# unzip database
unzip -d database -q database/3_variables_constants_7_nodes.txt.zip
```


## Pre-computed Simplification Lookup Tables

To generate an oracle, we need a simplification lookup table (or database) containing a large number of expressions. We used an enumerative search to pre-compute expressions with a bit size of 8, 16, 32 and 64 according to the following specifications:

* up to five variables `p0`, `p1`, `p2`, `p3` and `p4`

* truncation operators to downcast variables (if necessary) to 32, 16 and 8 bit,

* the bit vector operations addition, subtraction, multiplication, negation (unary minus), bitwise and/or/xor/not and the logical shift left,

* and, for some tables, the constants 0x0, 0x1, 0x2, 0x80, 0xff, 0x800, 0xffff, 0x8000_0000, 0xffff_ffff, 0x8000_0000_0000_0000 and 0xffff_ffff_ffff_ffff.

The example database included in [database](/database/) contains all 1,293,020 combinations created by using three variables and the constants 0x0, 0x1 and 0x2 for up to 7 nodes (e.g., `((p0 + p1) * (p2 ^ 0x2))` or `((p0 - p2) << (p1 + p2))`). Larger pre-computed databases can be found [here](https://synthesis.to/code/simplification_databases.7z) (~31GB unzipped). Note that the code for pre-computing expressions is __not__ part of this repository. We plan to release it at some point in the future.

## Stochastic Program Synthesis

As an alternative to pre-computed lookup tables, msynth supports expression simplification via stochastic program synthesis. For a given complex arithmetic expression, msynth can learn a shorter expression that shares the same input-output behavior. For now, it is implemented as a stand-alone component. However, we plan to combine both simplification approaches in future.

## Example Usage

First, let's generate a simplification oracle that uses a pre-computed simplification database as input and clusters the contained expressions into equivalence classes.

```
$ python scripts/gen_oracle.py database/3_variables_constants_7_nodes.txt oracle.pickle
msynth - INFO: Computing oracle for 30 variables and 50 samples. 
               Using library at 'database/3_variables_constants_7_nodes.txt'
msynth - INFO: Writing oracle to oracle.pickle
msynth - INFO: Done in 632.84 seconds
```

Depending on the size of the pre-computed simplification database, this may take a few minutes or hours, depending on your computer. Alternatively, you can use the pre-computed [oracle.pickle](/oracle.pickle).

Afterward, the serialized oracle can be used to simplify complex expressions:

```python
from msynth import Simplifier

# initialize simplifier
simplifier = Simplifier(oracle_path)
# simplify expression
simplified = simplifier.simplify(expression)
```

Alternatively, we can simplify complex expressions via program synthesis and learn expressions with the same input-output behavior:

```python
from msynth import Synthesizer

# initialize synthesizer
synthesizer = Synthesizer()
# simplify via program synthesis
simplified = synthesizer.simplify(expression)
```

It is also possible to combine expression simplification with Miasm's symbolic execution engine:

```
$ python scripts/symbolic_simplification.py samples/mba_challenge 0x1290 oracle.pickle
[snip]
before: {({RDI[0:32] & RSI[0:32] 0 32, 0x0 32 64} * 0x2 + {RDI[0:32] ^ RSI[0:32] 0 32, 0x0 32 64})[0:32] 0 32, 0x0 32 64}

msynth.simplifier - INFO: initial ast: {({RDI[0:32] & RSI[0:32] 0 32, 0x0 32 64} * 0x2 + {RDI[0:32] ^ RSI[0:32] 0 32, 0x0 32 64})[0:32] 0 32, 0x0 32 64}

msynth.simplifier - INFO: simplified subtree: ({RDI[0:32] & RSI[0:32] 0 32, 0x0 32 64} * 0x2 + {RDI[0:32] ^ RSI[0:32] 0 32, 0x0 32 64})[0:32] -> RDI[0:32] + RSI[0:32]

simplified: {RDI[0:32] + RSI[0:32] 0 32, 0x0 32 64}
[snip]
```

Further example usages can be found in the [scripts](/scripts) directory.


## Limitations and Future Work

* synthesis of partial constants not supported
* limited support for truncations and zero/sign extensions
* pre-compution tables might not be complete

## Contact

For more information, contact Tim Blazytko ([@mr_phrazer](https://twitter.com/mr_phrazer)) or Moritz Schloegel ([@m_u00d8](https://twitter.com/m_u00d8)).
