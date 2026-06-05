"""
SiMBA preprocessing pass for linear mixed Boolean-arithmetic expressions.

This is a Miasm-native reimplementation of the algorithm described in:
Benjamin Reichenwallner and Peter Meerwald-Stadler,
"Efficient Deobfuscation of Linear Mixed Boolean-Arithmetic Expressions",
CheckMATE 2022, DOI 10.1145/3560831.3564256, arXiv:2209.06335.

Reference implementation: https://github.com/DenuvoSoftwareSolutions/SiMBA
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from miasm.expression.expression import (
    Expr,
    ExprCompose,
    ExprCond,
    ExprId,
    ExprInt,
    ExprMem,
    ExprOp,
    ExprSlice,
)

from msynth.simplification._bitwise_table import (
    _MAX_TABLE_VARS,
    instantiate_recipe,
    minimal_bitwise_recipe,
    minimal_bitwise_recipes,
)
from msynth.utils.expr_utils import get_subexpressions

# Cap on the number of distinct nonzero signature values for which the
# decomposition search enumerates every coefficient subset. Linear MBAs in the
# wild have only a handful of distinct coefficients, so this is rarely hit; above
# it the search falls back to small (<=4) coefficient sets to stay cheap.
_MAX_DECOMP_VALUES = 8


class _ExpressionKind(Enum):
    """Coarse expression classes used by the linear-MBA validator."""

    ARITHMETIC = "arithmetic"
    BITWISE = "bitwise"
    MIXED = "mixed"


# Primary leaves: nodes that SiMBA always treats as opaque atoms because
# they are syntactic dead-ends with no "interior" structure for the
# linear-MBA classifier to descend through. ExprCond joins this set
# under the atomisation extension (GAMBA Section 5.5) — its value on a
# boolean cube assignment is deterministic and structurally identified,
# so the same soundness sketch as ExprSlice/ExprCompose carries over.
_PRIMARY_LEAVES = (ExprId, ExprMem, ExprSlice, ExprCompose, ExprCond)


def _expr_node_count(expr: Expr) -> int:
    """
    Cheap structural size of an :class:`Expr`. Used as the net-shrink guard
    for :func:`_bitwise_refine`, which only accepts a refined output when it
    has strictly fewer nodes than the input.
    """
    try:
        return len(expr.graph().nodes())
    except Exception:
        # ``Expr.graph()`` can fail on degenerate inputs (e.g. a bare leaf);
        # fall back to a conservative size of 1 — the refine guard then
        # rejects anything that isn't a strict improvement.
        return 1


def _bitwise_refine(expr: Expr) -> Expr:
    """
    Algebraic polish on the fully-assembled SimBA reconstruction.

    Reuses msynth's GAMBA §5.2 no-grow preprocessor (idempotence, De Morgan,
    absorption, redundancy, complement-pair, XOR-collapses, …) to refine the
    final SimBA output. Corresponds in spirit to upstream GAMBA's
    ``BitwiseFactory.refine`` (XOR-insertion · negation-flipping · common-
    factor extraction) — the no-grow §5.2 rules cover the negation-flip and
    XOR-collapse cases. Only ``GAMBA_PREPROCESSOR``'s ``guarded=False`` rules
    run: the guarded ``ring_normalize`` / ``factor_common_subterm`` rules are
    deliberately excluded so this stays a pure no-grow polish (those broader
    rewrites are the post-rewriter's job in GAMBA mode).

    Called once on the complete reconstruction at the end of
    :meth:`_SimbaSimplifier.simplify` — NOT per Quine-McCluskey region. Applying
    it per region perturbs SimBA's multi-coefficient assembly (the open issue
    that previously kept it disabled); applying it once to the assembled output
    operates on the whole coefficient-bearing expression and is sound. Each
    constituent rule is individually Z3-verified in
    ``tests/simplification/test_rewrites.py``, and the end-to-end soundness of
    this call is gated by ``test_simba_bitwise_refine_is_sound`` /
    ``test_simba_atoms``.

    The preprocessor is no-grow by construction; the net-shrink guard below is
    belt-and-braces against the rare case where bottom-up normalisation rebuilds
    equal-size nodes differently.

    Args:
        expr: The assembled SimBA reconstruction (any Expr).

    Returns:
        The refined expression when it has strictly fewer nodes than
        ``expr``; otherwise ``expr`` unchanged.
    """
    # Lazy import to avoid a top-of-module circular concern — ``simba`` is
    # imported by ``pipeline``, and ``gamba`` is also imported by
    # ``pipeline``; nothing in ``gamba`` depends on ``simba``, so the lazy
    # import is purely defensive (the dependency graph is currently acyclic
    # but a future change could break that quietly).
    from msynth.simplification.gamba import GAMBA_PREPROCESSOR

    refined = GAMBA_PREPROCESSOR.normalize(expr)
    if _expr_node_count(refined) < _expr_node_count(expr):
        return refined
    return expr


class _DeterministicRandom:
    """Tiny seeded LCG for reproducible 64-bit probe values (no global state)."""

    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next64(self) -> int:
        # SplitMix64-style mixing — good spread, fully deterministic.
        self._state = (self._state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _collect_terminals(expr: Expr) -> set:
    """Collect the opaque integer leaves of ``expr`` (descending through ops)."""
    leaves: set = set()

    def walk(node: Expr) -> None:
        if isinstance(node, ExprOp):
            for arg in node.args:
                walk(arg)
        elif isinstance(node, _PRIMARY_LEAVES):
            leaves.add(node)

    walk(expr)
    return leaves


def _eval_expr_int(expr: Expr, env: dict, mask: int) -> int:
    """Evaluate ``expr`` over full-width integers under ``env`` (terminals→int)."""
    if isinstance(expr, ExprInt):
        return int(expr) & mask
    if isinstance(expr, _PRIMARY_LEAVES):
        return env[expr] & mask
    if not isinstance(expr, ExprOp):
        raise ValueError(f"unsupported expression {type(expr).__name__}")
    args = [_eval_expr_int(arg, env, mask) for arg in expr.args]
    op = expr.op
    if op == "+":
        return sum(args) & mask
    if op == "-":
        if len(args) == 1:
            return (-args[0]) & mask
        result = args[0]
        for value in args[1:]:
            result -= value
        return result & mask
    if op == "*":
        result = 1
        for value in args:
            result *= value
        return result & mask
    if op == "&":
        result = mask
        for value in args:
            result &= value
        return result & mask
    if op == "|":
        result = 0
        for value in args:
            result |= value
        return result & mask
    if op == "^":
        result = 0
        for value in args:
            result ^= value
        return result & mask
    if op == "<<":
        shift = args[1] & (mask.bit_length() - 1)
        return (args[0] << shift) & mask
    if op == ">>":
        shift = args[1] & (mask.bit_length() - 1)
        return (args[0] >> shift) & mask
    raise ValueError(f"unsupported operation {op!r}")


def _apply_op_rule(
    op: str, args: tuple, arg_kinds: list, parent_size: int
) -> _ExpressionKind | None:
    """
    Per-op linear-MBA classification rule.

    Given an ExprOp's operator string, args, and the classified kinds
    of those args, return the kind of the whole op — or ``None`` if no
    rule in the linear-MBA fragment matches this operand-kind
    combination. The caller (:func:`_classify_uncached`) treats a
    ``None`` here as a whole-expression no-op signal (``(None, True)``),
    NOT as an instruction to atomise the node — operand-kind rejections
    are deliberately left unsimplified rather than turned into atoms.
    """
    mask = (1 << parent_size) - 1

    if op == "-" and len(args) == 1:
        return (
            _ExpressionKind.ARITHMETIC
            if arg_kinds[0] is _ExpressionKind.ARITHMETIC
            else _ExpressionKind.MIXED
        )

    if op in {"+", "-"} and len(args) >= 2:
        return (
            _ExpressionKind.ARITHMETIC
            if all(k is _ExpressionKind.ARITHMETIC for k in arg_kinds)
            else _ExpressionKind.MIXED
        )

    if op == "*" and len(args) >= 2:
        # Linear MBA terms can be multiplied by arithmetic constants, but a
        # product of two variable-dependent bitwise/mixed expressions would
        # be polynomial/nonlinear. The caller atomises that case.
        non_arithmetic = sum(k is not _ExpressionKind.ARITHMETIC for k in arg_kinds)
        if non_arithmetic > 1:
            return None
        return (
            _ExpressionKind.ARITHMETIC if non_arithmetic == 0 else _ExpressionKind.MIXED
        )

    if op in {"&", "|"} and len(args) >= 2:
        if all(k is _ExpressionKind.ARITHMETIC for k in arg_kinds):
            return _ExpressionKind.ARITHMETIC
        if all(k is _ExpressionKind.BITWISE for k in arg_kinds):
            return _ExpressionKind.BITWISE
        return None

    if op == "^" and len(args) >= 2:
        # XOR sits inside the linear-MBA fragment only under tight
        # conditions. The cube reconstruction extrapolates from
        # boolean-cube samples to all bit-vector inputs assuming the
        # underlying function is a linear MBA; classifying outside
        # that fragment produces rewrites that agree on {0,1}^n and
        # diverge elsewhere.
        #
        # Valid shapes (each preserves linear-MBA-ness):
        #   - all operands BITWISE  (possibly with all-ones constants
        #     standing in for bitwise NOT)               -> BITWISE
        #   - exactly one MIXED operand, the rest all-ones constants
        #     (this is ``~MIXED`` = ``-MIXED - 1``)      -> MIXED
        #   - all operands ARITHMETIC constants
        #     (XOR of constants is itself a constant)    -> ARITHMETIC
        # Everything else is non-linear in the operands and is left
        # to the caller to atomise.
        bitwise_count = 0
        mixed_count = 0
        allones_count = 0
        non_allones_arith_count = 0
        for arg, kind in zip(args, arg_kinds):
            if kind is _ExpressionKind.BITWISE:
                bitwise_count += 1
            elif kind is _ExpressionKind.MIXED:
                mixed_count += 1
            else:
                if isinstance(arg, ExprInt) and int(arg) == mask:
                    allones_count += 1
                else:
                    non_allones_arith_count += 1

        if bitwise_count == 0 and mixed_count == 0:
            return _ExpressionKind.ARITHMETIC
        if non_allones_arith_count > 0:
            return None
        if bitwise_count > 0 and mixed_count > 0:
            return None
        if mixed_count == 0:
            return _ExpressionKind.BITWISE
        if mixed_count == 1:
            return _ExpressionKind.MIXED
        return None

    return None


def _classify(
    expr: Expr,
    parent_size: int,
    cache: dict[Expr, tuple[_ExpressionKind | None, bool]] | None = None,
) -> tuple[_ExpressionKind | None, bool]:
    """
    Atomisation-aware classifier for SiMBA.

    Returns ``(kind, is_atom)`` where ``kind`` is the linear-MBA kind
    (ARITHMETIC, BITWISE, or MIXED) the surrounding cube reasoning sees
    for this node, and ``is_atom`` records whether SiMBA treats the
    node as opaque (i.e. looks it up in the cube ``env`` rather than
    recursing into its structure).

    The atomisation extension (GAMBA Section 5.5) generalises the
    Slice/Compose/Mem leaves to every node whose *operator* is outside
    the linear-MBA fragment (the fast path in
    :func:`_classify_uncached`): such a node is returned as a BITWISE
    atom. Soundness rests on three properties that hold for every miasm
    pure-function node:

    1. Determinism per cube assignment — when the inner atoms take
       fixed integer values, the node takes a deterministic integer
       value.
    2. Structural dedup — two textually equal occurrences map to the
       same atom via miasm's ``Expr.__hash__`` / ``__eq__``.
    3. Width match — checked here via ``expr.size != parent_size``,
       so ``env[node]`` and the cube modulus align.

    Note the distinction from an *operand-kind* rejection: when the
    operator IS in the fragment but its operand kinds don't match any
    rule, the classifier does NOT atomise — it returns the no-op signal
    ``(None, True)`` instead (see :func:`_classify_uncached`).

    ``(None, True)`` (the no-op short-circuit signal used by callers) is
    returned in three cases: a fundamental width mismatch, an operand-kind
    rejection, and propagation of either from a child argument. It is NOT
    limited to width mismatches.
    """
    if cache is None:
        cache = {}
    cached = cache.get(expr)
    if cached is not None:
        return cached

    result = _classify_uncached(expr, parent_size, cache)
    cache[expr] = result
    return result


def _classify_uncached(
    expr: Expr,
    parent_size: int,
    cache: dict[Expr, tuple[_ExpressionKind | None, bool]],
) -> tuple[_ExpressionKind | None, bool]:
    if expr.size != parent_size:
        return (None, True)
    if isinstance(expr, ExprInt):
        return (_ExpressionKind.ARITHMETIC, False)
    if isinstance(expr, _PRIMARY_LEAVES):
        return (_ExpressionKind.BITWISE, True)
    if not isinstance(expr, ExprOp):
        # Any other node kind (future miasm IL extensions) is treated
        # as an opaque BITWISE atom. Soundness is the same as for the
        # primary leaves above.
        return (_ExpressionKind.BITWISE, True)

    # Fast path: if the op string is outside the linear-MBA fragment,
    # atomise the whole node without recursing into its args. This
    # both saves work and avoids spurious None-propagation when an op
    # like ``<<`` has args whose width differs from the op result —
    # the cube reasoning doesn't look inside, so the inner widths are
    # irrelevant.
    if expr.op not in {"+", "-", "*", "&", "|", "^"}:
        return (_ExpressionKind.BITWISE, True)

    # Genuine product of two or more non-constant factors (``A·B``, ``x·y``) is a
    # nonlinear term. Atomise it (GAMBA Section 5.5) so the SURROUNDING linear
    # MBA still reconstructs, with the product treated as one opaque cube value.
    # A constant-scaled linear term ``c·expr`` keeps its single non-constant
    # factor and is handled by the linear ``*`` rule below, so it is NOT caught
    # here. Unlike the ``&``/``|``/``^`` operand-kind rejections, atomising a
    # multiplication is exactly the §5.5 case and keeps the product verbatim in
    # the output, so it never widens the atom set in a way the pipeline cannot
    # fold back.
    if expr.op == "*":
        non_constant = [arg for arg in expr.args if not isinstance(arg, ExprInt)]
        if len(non_constant) >= 2:
            return (_ExpressionKind.BITWISE, True)

    arg_results = [_classify(arg, parent_size, cache) for arg in expr.args]
    if any(k is None for k, _ in arg_results):
        # An arg has a fundamental width mismatch or an unrecoverable
        # operand-kind rejection deeper inside — propagate the no-op
        # signal. (See the "operand-kind rejection" note below for
        # why we keep this strict.)
        return (None, True)
    arg_kinds = [k for k, _ in arg_results]

    op_kind = _apply_op_rule(expr.op, expr.args, arg_kinds, parent_size)
    if op_kind is None:
        # Operand-kind rejection: the op IS in {+, -, *, &, |, ^} but
        # this particular operand-kind combination (e.g. ``&`` over
        # one BITWISE and one MIXED arg) doesn't match any linear-MBA
        # rule. We deliberately DO NOT atomise here.
        #
        # GAMBA 5.5's "substitution of nonlinear subexpressions" is
        # about replacing whole non-linear *operators* (shifts, cond,
        # division, etc.) with opaque atoms; that's handled by the
        # fast path above. Atomising operand-kind rejections is a
        # different beast — it widens SiMBA's atom set whenever a
        # linear-MBA-shaped op happens to mix kinds at depth, which
        # in practice triggers verbose reconstructions over many
        # atoms whose downstream the surrounding pipeline cannot
        # fold back together (the demo MBA regression).
        #
        # Returning ``None`` here keeps SiMBA's preprocessor pass a
        # no-op on those shapes, matching pre-extension behaviour.
        return (None, True)
    return (op_kind, False)


def _collect_atoms(expr: Expr) -> list[Expr]:
    """
    Collect the leaf expressions SiMBA treats as boolean-cube atoms.

    Under the atomisation extension (GAMBA Section 5.5), an atom is
    any node where :func:`_classify` reports ``is_atom=True`` — either
    a primary leaf (``ExprId``, ``ExprMem``, ``ExprSlice``,
    ``ExprCompose``, ``ExprCond``) or any subtree whose op + arg-kinds
    don't match a linear-MBA rule. The walker stops descending at
    those points and adds the whole subtree as a single atom.

    Structural identity matters: two textually equal subtrees must
    dedupe to one atom, or ``e ^ e`` would not collapse to zero.
    Miasm's ``Expr.__hash__`` / ``__eq__`` are structural, which makes
    the ``set`` here do the right thing for nested ExprOps as well as
    for the primary leaves.
    """
    parent_size = expr.size
    cache: dict[Expr, tuple[_ExpressionKind | None, bool]] = {}
    atoms: set[Expr] = set()

    def walk(node: Expr) -> None:
        kind, is_atom = _classify(node, parent_size, cache)
        if kind is None:
            # No-op signal (width mismatch or operand-kind rejection):
            # the node is outside the fragment and is not atomised.
            return
        if isinstance(node, ExprInt):
            return
        if is_atom:
            atoms.add(node)
            return
        # Decomposable: recurse into args.
        if isinstance(node, ExprOp):
            for arg in node.args:
                walk(arg)

    walk(expr)
    return sorted(atoms, key=lambda x: str(x))


# ---------------------------------------------------------------------------
# Quine-McCluskey boolean minimisation
# ---------------------------------------------------------------------------
#
# Used by ``_SimbaSimplifier._lookup_bitwise_expression`` to turn a truth
# table into a minimal sum-of-products. Output is a list of (value, mask)
# tuples where bits in ``mask`` are don't-cares and the remaining bits in
# ``value`` are the required literal values. Round-trips through
# ``_build_qm_bitwise`` to a bitwise miasm expression.


def _qm_minimise(table: int, n_vars: int) -> list[tuple[int, int]]:
    """
    Quine-McCluskey minimisation of a boolean truth table.

    ``table`` packs row i of the function in bit i (matching
    ``_SimbaSimplifier._table_to_int``). ``n_vars`` is the variable
    count, so the table has ``2 ** n_vars`` rows. Returns a list of
    prime implicants chosen by an essential-implicant + greedy cover
    of the original minterms.

    Edge cases:
      - ``table == 0`` -> ``[]`` (the function is constant 0).
      - all-1s -> ``[(0, full_mask)]`` (single don't-care-everything term,
        meaning constant 1).
    """
    if table == 0:
        return []
    full_mask = (1 << n_vars) - 1
    rows = 1 << n_vars
    if table == (1 << rows) - 1:
        return [(0, full_mask)]

    # Collect minterms (rows where the function is 1).
    minterms = [row for row in range(rows) if (table >> row) & 1]

    # Group implicants by popcount of the "value" (don't-care bits excluded).
    # Each implicant is (value, mask). Iteratively combine pairs that
    # differ in exactly one literal; the combined implicant gains a
    # don't-care bit. Implicants that participate in a combination are
    # marked, and the unmarked ones at every round are prime implicants.
    current: set[tuple[int, int]] = {(m, 0) for m in minterms}
    primes: set[tuple[int, int]] = set()

    while current:
        # Bucket by mask so we only attempt to combine same-shape implicants
        # (different-shape pairs can't differ in exactly one literal).
        by_mask: dict[int, list[tuple[int, int]]] = {}
        for value, mask in current:
            by_mask.setdefault(mask, []).append((value, mask))

        used: set[tuple[int, int]] = set()
        next_round: set[tuple[int, int]] = set()
        for mask, items in by_mask.items():
            # Within a single mask group, bucket further by popcount of
            # the literal bits so a pair that differs in one bit lies
            # exactly between adjacent groups.
            by_popcount: dict[int, list[tuple[int, int]]] = {}
            for value, m in items:
                literal_bits = value & ~mask
                by_popcount.setdefault(bin(literal_bits).count("1"), []).append(
                    (value, m)
                )
            popcounts = sorted(by_popcount.keys())
            for pc in popcounts:
                if pc + 1 not in by_popcount:
                    continue
                for value_a, mask_a in by_popcount[pc]:
                    for value_b, mask_b in by_popcount[pc + 1]:
                        diff = (value_a ^ value_b) & ~mask
                        if diff and (diff & (diff - 1)) == 0:
                            combined_mask = mask | diff
                            combined_value = value_a & ~combined_mask
                            next_round.add((combined_value, combined_mask))
                            used.add((value_a, mask_a))
                            used.add((value_b, mask_b))

        # Anything not combined this round is a prime implicant.
        for item in current:
            if item not in used:
                primes.add(item)
        current = next_round

    prime_list = sorted(primes)

    # Cover step: select essential primes first, then greedily cover the
    # rest. ``covers[p]`` is the set of original minterms covered by p.
    def implicant_covers(value: int, mask: int) -> set[int]:
        # Enumerate every assignment of the don't-care bits.
        dc_bits = [b for b in range(n_vars) if (mask >> b) & 1]
        covered: set[int] = set()
        for combo in range(1 << len(dc_bits)):
            cv = value
            for i, b in enumerate(dc_bits):
                if (combo >> i) & 1:
                    cv |= 1 << b
                else:
                    cv &= ~(1 << b)
            covered.add(cv)
        return covered

    covers = {p: implicant_covers(*p) for p in prime_list}
    minterm_set = set(minterms)

    # For each minterm, which primes cover it?
    cover_map: dict[int, list[tuple[int, int]]] = {m: [] for m in minterm_set}
    for p, mts in covers.items():
        for mt in mts:
            if mt in cover_map:
                cover_map[mt].append(p)

    chosen: set[tuple[int, int]] = set()
    remaining = set(minterm_set)

    # Essential prime implicants: those that are the unique cover of
    # some minterm. Repeatedly extract them until none remain — picking
    # one essential may leave others as essential by simplification.
    while True:
        new_essentials = {
            cover_map[mt][0]
            for mt in remaining
            if len(cover_map[mt]) == 1 and cover_map[mt][0] not in chosen
        }
        if not new_essentials:
            break
        for p in new_essentials:
            chosen.add(p)
            remaining -= covers[p]
        # Drop chosen primes from cover_map so further iterations see
        # the reduced choice set.
        for mt in list(remaining):
            cover_map[mt] = [p for p in cover_map[mt] if p not in chosen]

    # Greedy cover for whatever the essentials didn't pick up.
    while remaining:
        best = max(prime_list, key=lambda p: len(covers[p] & remaining))
        if not (covers[best] & remaining):
            # Should not happen if the prime set is complete, but bail
            # gracefully rather than loop forever.
            break
        chosen.add(best)
        remaining -= covers[best]

    return sorted(chosen)


@dataclass(frozen=True)
class SimbaPass:
    """Simplify supported linear MBAs before oracle-backed simplification."""

    name: str = "simba"

    def run(self, expr: Expr) -> Expr:
        simplifier = _SimbaSimplifier(expr)
        return simplifier.simplify()


# Bound on how deep the bottom-up nested-MBA simplification recurses. Real
# corpus expressions nest only a handful of linear-MBA layers; the cap is a
# safety net against pathological inputs.
_MAX_BOTTOM_UP_DEPTH = 12

# Cap on the number of boolean-cube atoms SiMBA will reconstruct over. The cube
# machinery is exponential in the atom count (``_signature`` is 2^n and
# ``_generic_linear_combination`` is O(4^n)), so atomising many independent
# products (e.g. a sum of a dozen ``bitwise·bitwise`` monomials) would blow up.
# Linear MBAs that actually simplify to a compact form use only a handful of
# atoms, so bailing above this bound costs no coverage and keeps the pass fast.
_MAX_CUBE_ATOMS = 8

# Bound on bottom-up recursion input size, so a huge nested expression does not
# trigger an expensive cascade of operand re-simplifications.
_MAX_BOTTOM_UP_NODES = 120


class _SimbaSimplifier:
    def __init__(self, expr: Expr, depth: int = 0):
        self.expr = expr
        self.size = expr.size
        self.modulus = 1 << self.size
        self.mask = self.modulus - 1
        self._classify_cache: dict[Expr, tuple[_ExpressionKind | None, bool]] = {}
        self.variables = _collect_atoms(expr)
        self._depth = depth

    def simplify(self) -> Expr:
        # Soundness gate: SiMBA's cube reconstruction is only valid for genuine
        # linear MBAs, and the atomisation / bottom-up extensions can, on rare
        # adversarial shapes, reconstruct a sub-expression to two inconsistent
        # atom forms and produce a non-equivalent result. Verify every non-trivial
        # output against the input on edge-case + random probes and fall back to
        # the (always-correct) input when verification fails, so the pass can only
        # ever return an equivalent expression.
        candidates = [self.expr]
        core = self._simplify_core()
        if core is not self.expr:
            candidates.append(core)
        # At the top level also try reconstructing the whole expression as a
        # linear MBA over its bare variables. This recovers the compact form of
        # heavily-obfuscated 2-3 variable expressions whose arithmetic-inside-
        # bitwise shape (``(x-y) & y``, ``(-x) & y``, …) the structural classifier
        # rejects, but which are still simple functions of x and y.
        if self._depth == 0:
            bare = self._simplify_over_bare_variables()
            if bare is not None and bare is not self.expr:
                candidates.append(bare)
        verified = [self.expr] + [
            candidate
            for candidate in candidates[1:]
            if self._verify_reconstruction(candidate)
        ]
        return min(verified, key=_expr_node_count)

    def _simplify_over_bare_variables(self) -> Expr | None:
        """Reconstruct the whole expression over its bare variables, or None.

        Treats the bare terminals (``ExprId``/``ExprMem``/…) as the cube atoms and
        evaluates the *entire* expression numerically on the Boolean cube, then
        runs the standard signature reconstruction. Sound only when the
        expression is a linear MBA of those variables — the outer soundness gate
        verifies the result and discards it otherwise, so this can be tried
        aggressively.
        """
        bare = sorted(_collect_terminals(self.expr), key=repr)
        if not bare or len(bare) > _MAX_CUBE_ATOMS:
            return None
        try:
            signature = []
            for assignment in range(1 << len(bare)):
                env = {
                    variable: (assignment >> index) & 1
                    for index, variable in enumerate(bare)
                }
                signature.append(_eval_expr_int(self.expr, env, self.mask))
        except (KeyError, ValueError, OverflowError):
            return None
        if len(set(signature)) == 1:
            return self._const(signature[0])
        try:
            reconstructed = self._simplify_signature(signature, bare)
        except (KeyError, ValueError, OverflowError):
            return None
        return _bitwise_refine(reconstructed)

    def _verify_reconstruction(self, simplified: Expr) -> bool:
        """True iff ``simplified`` agrees with the input on edge + random probes."""
        terminals = sorted(
            _collect_terminals(self.expr) | _collect_terminals(simplified), key=repr
        )
        # Deterministic edge cases (catch modular-wraparound / bit-pattern bugs)
        # followed by random probes.
        edge = [0, 1, 2, 3, self.mask, self.mask - 1, self.modulus >> 1, 0xFF, 0x80]
        rng = _DeterministicRandom(0x511B7A)
        for trial in range(56):
            if trial < len(edge):
                value = edge[trial]
                env = {term: value for term in terminals}
            else:
                env = {term: rng.next64() for term in terminals}
            try:
                if _eval_expr_int(self.expr, env, self.mask) != _eval_expr_int(
                    simplified, env, self.mask
                ):
                    return False
            except (KeyError, ValueError):
                # An operator the lightweight evaluator does not model — trust the
                # reconstruction (these shapes are atomised, not reconstructed).
                return True
        return True

    def _simplify_core(self) -> Expr:
        if self.size <= 0:
            return self.expr

        # Collapse polynomial obfuscation of products before classification:
        # ``(A&B)·(A|B) + (A&~B)·(~A&B) == A·B`` rewrites a sum of two bitwise
        # products into a single ``A·B`` factor. SiMBA then treats ``A·B`` as an
        # opaque atom and reconstructs the surrounding linear MBA normally, so a
        # polynomial MBA like ``(x&y)·(x|y)+(x&~y)·(~x&y)-20`` becomes ``x·y-20``.
        collapsed = self._collapse_product_identities(self.expr)
        if collapsed is not self.expr:
            self.expr = collapsed
            self.variables = _collect_atoms(collapsed)

        # A None here is the classifier's no-op signal: either a width
        # mismatch, or the top node is a linear-MBA operator whose operand
        # kinds don't match any rule (operand-kind rejection). Operators
        # outside the fragment instead atomise to a single opaque BITWISE
        # atom, which SiMBA reconstructs unchanged. Before giving up we try a
        # bottom-up pass: simplifying operand sub-expressions can collapse a
        # nested linear MBA (``B`` inside ``B & y``) to an atom and unblock this
        # node's reconstruction.
        if self._classify(self.expr) is None:
            return self._simplify_bottom_up()

        # Guard the exponential cube machinery: too many independent atoms (many
        # distinct products in a polynomial MBA) make signature/reconstruction
        # infeasible. Such inputs are left unchanged, exactly as the pre-
        # atomisation classifier did by rejecting products outright.
        if len(self.variables) > _MAX_CUBE_ATOMS:
            return self.expr

        try:
            # The core theorem in the paper says a linear MBA is determined by
            # its values on all Boolean assignments. We evaluate the whole Miasm
            # expression directly on those assignments instead of decomposing it
            # into coefficient/bitwise-expression terms first.
            signature = self._signature(self.expr, self.variables)
            simplified = self._simplify_signature(signature, self.variables)
        except (KeyError, ValueError, OverflowError):
            # Any unsupported Miasm form, missing variable binding, or arithmetic
            # edge case should make the pass transparent. Preprocessing must not
            # turn a valid expression into an exception for the simplifier.
            return self.expr

        if self._effective_variable_count(simplified) <= 3 and len(self.variables) > 3:
            # Generic reconstruction can eliminate variables. If that leaves a
            # small-variable expression, rerun SiMBA so the lookup/refinement path
            # for one, two, or three variables can produce a more compact result.
            simplified = self._simplify_fewer_variables(simplified)

        # Polish the FULLY-ASSEMBLED reconstruction with the §5.2 no-grow
        # algebraic rules (see :func:`_bitwise_refine`). Applied here — after
        # all coefficient/bitwise terms are combined — rather than per QM region,
        # so the rewrite operates on the complete coefficient-bearing expression
        # and cannot perturb SimBA's multi-coefficient assembly. Net-shrink
        # guarded, so it only ever makes the output smaller. This keeps SimBA's
        # standalone output compact in SIMBA mode and in subtree-SimBA, where no
        # GAMBA post-rewriter runs after the pass.
        return _bitwise_refine(simplified)

    def _simplify_bottom_up(self) -> Expr:
        """Simplify operand sub-expressions, then retry this node.

        When the node itself is outside the linear-MBA fragment (e.g. ``B & y``
        where ``B`` is a linear MBA, not an atom), reducing ``B`` to its compact
        form ``a`` turns the node into ``a & y`` — a valid bitwise-over-atoms
        shape SiMBA can reconstruct. Each operand is reduced independently and
        only kept when it does not grow, so the rewrite is sound and never
        regresses. Bounded by ``_MAX_BOTTOM_UP_DEPTH``.
        """
        if (
            self._depth >= _MAX_BOTTOM_UP_DEPTH
            or not isinstance(self.expr, ExprOp)
            or _expr_node_count(self.expr) > _MAX_BOTTOM_UP_NODES
        ):
            return self.expr
        new_args = []
        changed = False
        for arg in self.expr.args:
            if isinstance(arg, ExprOp) and arg.size == self.size:
                reduced = _SimbaSimplifier(arg, depth=self._depth + 1).simplify()
                if reduced != arg and _expr_node_count(reduced) <= _expr_node_count(
                    arg
                ):
                    new_args.append(reduced)
                    changed = True
                    continue
            new_args.append(arg)
        if not changed:
            return self.expr
        rebuilt = ExprOp(self.expr.op, *new_args)
        # Retry the full reconstruction on the operand-reduced node. Its operands
        # are now fixpoints, so a second bottom-up pass cannot loop.
        return _SimbaSimplifier(rebuilt, depth=self._depth + 1).simplify()

    def _classify(self, expr: Expr) -> _ExpressionKind | None:
        """
        Return the linear-MBA kind of ``expr`` or None (the no-op
        signal) when ``expr`` is outside the cube model.

        Under the atomisation extension, None is returned for a size
        mismatch with ``self.size`` OR for an operand-kind rejection (a
        fragment operator whose operand kinds match no rule). Operators
        outside the linear-MBA fragment — shifts, rotations, division,
        multiplication of two non-arith operands, ExprCond — do NOT
        return None: they classify as BITWISE because the cube reasoning
        treats them as opaque atoms. See the module-level
        :func:`_classify` for the soundness sketch.
        """
        kind, _ = _classify(expr, self.size, self._classify_cache)
        return kind

    def _is_atom(self, expr: Expr) -> bool:
        """True iff SiMBA treats ``expr`` as opaque on the cube."""
        _, is_atom = _classify(expr, self.size, self._classify_cache)
        return is_atom

    def _is_all_ones(self, expr: Expr) -> bool:
        return isinstance(expr, ExprInt) and int(expr) == self.mask

    def _signature(self, expr: Expr, variables: list[Expr]) -> list[int]:
        """
        Evaluate ``expr`` on the Boolean cube for the sorted variable list.

        Assignment index bits encode variable values, matching the ordering used
        in the SiMBA paper and upstream code:
        0 -> (0, 0, ...), 1 -> (1, 0, ...), 2 -> (0, 1, ...), etc.
        """
        values = []
        for assignment in range(1 << len(variables)):
            env = {
                variable: (assignment >> index) & 1
                for index, variable in enumerate(variables)
            }
            values.append(self._evaluate(expr, env))
        return values

    def _evaluate(self, expr: Expr, env: dict[Expr, int]) -> int:
        """Evaluate the supported linear-MBA fragment under one Boolean assignment."""
        if isinstance(expr, ExprInt):
            return int(expr) & self.mask
        if self._is_atom(expr):
            # Primary leaf or atomised non-linear subtree — the cube
            # treats it as an opaque variable and looks it up directly.
            return env[expr] & self.mask
        if not isinstance(expr, ExprOp):
            raise ValueError(f"unsupported expression {type(expr).__name__}")

        args = [self._evaluate(arg, env) for arg in expr.args]
        if expr.op == "-" and len(args) == 1:
            return (-args[0]) & self.mask
        if expr.op == "+":
            return sum(args) & self.mask
        if expr.op == "-" and len(args) >= 2:
            result = args[0]
            for arg in args[1:]:
                result -= arg
            return result & self.mask
        if expr.op == "*":
            result = 1
            for arg in args:
                result *= arg
            return result & self.mask
        if expr.op == "&":
            result = self.mask
            for arg in args:
                result &= arg
            return result & self.mask
        if expr.op == "|":
            result = 0
            for arg in args:
                result |= arg
            return result & self.mask
        if expr.op == "^":
            result = 0
            for arg in args:
                result ^= arg
            return result & self.mask

        raise ValueError(f"unsupported operation {expr.op!r}")

    def _simplify_signature(self, signature: list[int], variables: list[Expr]) -> Expr:
        if len(set(signature)) == 1:
            return self._const(signature[0])

        # The always-valid conjunction-basis representation is the floor: every
        # other candidate is compared against it and we keep the smallest, so the
        # result can never be worse than this.
        generic = self._generic_linear_combination(signature, variables)
        return self._minimal_linear_reconstruction(signature, variables, generic)

    def _minimal_linear_reconstruction(
        self, signature: list[int], variables: list[Expr], generic: Expr
    ) -> Expr:
        """
        Pick the smallest equivalent reconstruction of the signature.

        A linear MBA decomposes as ``const + Σ c_k · f_k`` where each ``f_k`` is a
        0/1 boolean function of the atoms. We enumerate candidate decompositions
        whose coefficients are drawn from the distinct nonzero signature values
        (after peeling the constant), build each region's predicate via the
        minimal bitwise table (≤3 vars) or the legacy lookup (more vars), and
        return whichever assembled expression has the fewest nodes — always
        including ``generic`` and the paper's ``_refine`` heuristics so the result
        is never larger than before.
        """
        candidates: list[Expr] = [generic]

        # Legacy small-variable heuristics (now table-backed) remain a useful
        # candidate source for the negated-predicate / value-merge shapes.
        if len(variables) <= _MAX_TABLE_VARS:
            try:
                candidates.append(self._refine(signature, variables, generic))
            except (ValueError, KeyError):
                pass

        const = signature[0] & self.mask
        shifted = [(value - const) & self.mask for value in signature]
        distinct = sorted({value for value in shifted if value != 0})
        for coeffs in self._iter_coeff_sets(distinct):
            candidates.extend(
                self._build_decomposition(shifted, variables, coeffs, const)
            )

        best = min(candidates, key=_expr_node_count)
        # Fold a leading constant into a negated factor when the shapes line up
        # (``c + c·f == (-c)·~f``). This recovers the reference's canonical
        # ``coeff · ~bitwise`` form, which is far more SMT-friendly than the
        # expanded ``c + c·f`` (the simplifier's own Z3 equivalence checks rely
        # on this) at a cost of at most one extra node.
        folded = self._fold_constant_into_negation(best)
        if (
            folded is not None
            and _expr_node_count(folded) <= _expr_node_count(best) + 1
        ):
            return folded
        return best

    def _fold_constant_into_negation(self, expr: Expr) -> Expr | None:
        """Rewrite ``c + c·f + …`` to ``(-c)·~f + …`` (one term), else None.

        Uses the modular identity ``c + c·f == (-c)·~f`` for a 0/1 predicate
        ``f``. Only fires when the standalone constant equals the coefficient of
        one of the product terms, which is exactly the affine shape the SiMBA
        reference emits as ``coeff · ~x``.
        """
        if not (isinstance(expr, ExprOp) and expr.op == "+"):
            return None
        const_value: int | None = None
        const_index = -1
        for index, term in enumerate(expr.args):
            if isinstance(term, ExprInt):
                const_value = int(term) & self.mask
                const_index = index
                break
        if not const_value:
            return None
        for index, term in enumerate(expr.args):
            if index == const_index:
                continue
            if (
                isinstance(term, ExprOp)
                and term.op == "*"
                and len(term.args) == 2
                and isinstance(term.args[0], ExprInt)
            ):
                coeff = int(term.args[0]) & self.mask
                base = term.args[1]
            elif isinstance(term, ExprOp) and term.op == "-" and len(term.args) == 1:
                # Unary negation is coefficient -1 (see :meth:`_multiply`).
                coeff = self.mask
                base = term.args[0]
            else:
                coeff = 1
                base = term
            if coeff != const_value:
                continue
            folded_term = self._multiply((-coeff) & self.mask, self._invert(base))
            remaining = [
                arg
                for position, arg in enumerate(expr.args)
                if position != const_index and position != index
            ]
            return self._sum([folded_term, *remaining])
        return None

    def _iter_coeff_sets(self, distinct: list[int]):
        """Yield coefficient subsets to try as the basis of a decomposition.

        Every shifted signature value must be a subset-sum of the chosen
        coefficients, so enumerating subsets of the distinct values covers the
        per-value, negated, and additive-merge decompositions. The number of
        distinct values can be large (a complex linear MBA has many distinct
        coefficients), so the enumeration is bounded: the full per-value set and
        every singleton are always tried, and the combinatorial pair/triple
        merges only when there are few distinct values.
        """
        count = len(distinct)
        if count == 0:
            return
        # Per-value decomposition (one term per distinct value).
        yield tuple(distinct)
        # Single-coefficient decompositions (``c·f`` and the negated/affine forms).
        for value in distinct:
            yield (value,)
        # Additive-merge decompositions (``a, b → a+b`` shared regions) — only
        # worth the combinatorial cost when the value set is small.
        if 2 <= count <= _MAX_DECOMP_VALUES:
            for size in (2, 3):
                if size >= count:
                    break
                for combo in combinations(distinct, size):
                    yield combo

    def _build_decomposition(
        self,
        shifted: list[int],
        variables: list[Expr],
        coeffs: tuple[int, ...],
        const: int,
    ) -> Expr | None:
        """Build ``const + Σ coeffs[i] · f_i`` for one coefficient set, or None.

        Each row value must be representable as a subset-sum of ``coeffs`` (with
        the all-zero row using the empty subset); the chosen subset assigns which
        ``f_i`` are true on that row. Returns None if a value is unreachable or a
        region's bitwise expression cannot be built.
        """
        count = len(coeffs)
        empty: list[Expr] = []
        # Map each achievable subset-sum to a representative subset bitmask,
        # preferring fewer coefficients (so a row equal to a single coefficient
        # uses just that coefficient's predicate).
        sum_to_mask: dict[int, int] = {}
        for bits in sorted(range(1 << count), key=lambda b: bin(b).count("1")):
            total = 0
            for index in range(count):
                if (bits >> index) & 1:
                    total = (total + coeffs[index]) & self.mask
            sum_to_mask.setdefault(total, bits)

        predicates = [[0] * len(shifted) for _ in range(count)]
        for row, value in enumerate(shifted):
            if value == 0:
                continue
            bits = sum_to_mask.get(value)
            if bits is None:
                return empty
            for index in range(count):
                if (bits >> index) & 1:
                    predicates[index][row] = 1

        # Build, for each coefficient, the set of equal-cost bitwise forms for
        # its region, then choose forms that share subexpressions across terms
        # (so e.g. two terms both reuse ``y^z``).
        active = [index for index in range(count) if any(predicates[index])]
        candidate_lists = []
        for index in active:
            cands = self._lookup_bitwise_all(predicates[index], variables)
            if not cands:
                return empty
            candidate_lists.append(cands)
        chosen = self._select_sharing(candidate_lists)
        pairs = [(coeffs[index], chosen[pos]) for pos, index in enumerate(active)]
        # Two assemblies, both valid; the caller keeps whichever is smaller:
        #   - signed: ``A - B`` / ``3·A - 3·B`` (binary minus, shared coefficient),
        #   - additive: ``Σ coeff·f`` which the negation-fold can turn into ``~f``.
        forms = [self._assemble_signed(const, pairs)]
        additive_terms: list[Expr] = []
        if const:
            additive_terms.append(self._const(const))
        additive_terms.extend(self._multiply(coeff, factor) for coeff, factor in pairs)
        forms.append(self._sum(additive_terms))
        return forms

    def _lookup_bitwise_all(
        self, predicate: list[int], variables: list[Expr]
    ) -> list[Expr]:
        """All equal-cost minimal bitwise forms for ``predicate`` (≤3-var support).

        Falls back to the single legacy form for larger supports.
        """
        table = self._table_to_int(predicate)
        if table == 0:
            return [self._const(0)]
        support = self._predicate_support(predicate, len(variables))
        if len(support) <= _MAX_TABLE_VARS:
            projected = self._project_predicate(predicate, support, len(variables))
            recipes = minimal_bitwise_recipes(projected, len(support))
            if recipes:
                proj_vars = [variables[index] for index in support]
                return [
                    instantiate_recipe(
                        recipe,
                        proj_vars,
                        self._invert,
                        self._conjunction,
                        self._or,
                        self._xor,
                    )
                    for recipe in recipes
                ]
        single = self._lookup_bitwise_expression(predicate, variables)
        return [single] if single is not None else []

    def _select_sharing(self, candidate_lists: list[list[Expr]]) -> list[Expr]:
        """Pick one bitwise form per term to maximise shared subexpressions.

        A subexpression that can appear in two or more terms' candidate sets is
        "shareable"; each term then prefers the candidate covering the most
        shareable subexpressions (tie-break: fewest nodes). Order-independent.
        """
        subsets = [
            [set(get_subexpressions(cand)) for cand in cands]
            for cands in candidate_lists
        ]
        appearances: dict = {}
        for term_sets in subsets:
            union = set().union(*term_sets) if term_sets else set()
            for sub in union:
                appearances[sub] = appearances.get(sub, 0) + 1
        shareable = {sub for sub, count in appearances.items() if count >= 2}
        chosen: list[Expr] = []
        for cands, term_sets in zip(candidate_lists, subsets):
            best_index = max(
                range(len(cands)),
                key=lambda i: (len(term_sets[i] & shareable), -len(term_sets[i])),
            )
            chosen.append(cands[best_index])
        return chosen

    def _assemble_signed(self, const: int, pairs: list) -> Expr:
        """Assemble ``const + Σ coeff·f`` using subtraction for negative terms.

        Splitting into positive and negative magnitudes lets the result use
        ``A - B`` / ``3·A - 3·B`` (binary minus, shared positive coefficient)
        instead of ``A + (-B)`` / ``A + 0xFF..FD·B``, matching the reference's
        compact, SMT-friendly shapes.
        """
        half = self.modulus >> 1
        positive: list[Expr] = []
        negative: list[Expr] = []

        def emit(magnitude: int, factor: Expr, bucket: list[Expr]) -> None:
            if magnitude == 1:
                bucket.append(factor)
            else:
                bucket.append(ExprOp("*", self._const(magnitude), factor))

        const &= self.mask
        if const:
            if const < half:
                positive.append(self._const(const))
            else:
                negative.append(self._const((self.modulus - const) & self.mask))
        for coeff, factor in pairs:
            coeff &= self.mask
            if coeff == 0:
                continue
            if coeff < half:
                emit(coeff, factor, positive)
            else:
                emit((self.modulus - coeff) & self.mask, factor, negative)

        if not positive and not negative:
            return self._const(0)
        if not negative:
            return positive[0] if len(positive) == 1 else ExprOp("+", *positive)
        negative_combined = negative if len(negative) >= 1 else []
        if not positive:
            # All-negative: -(Σ negative).
            inner = negative[0] if len(negative) == 1 else ExprOp("+", *negative)
            return ExprOp("-", inner)
        positive_combined = (
            positive[0] if len(positive) == 1 else ExprOp("+", *positive)
        )
        return ExprOp("-", positive_combined, *negative_combined)

    def _generic_linear_combination(
        self, signature: list[int], variables: list[Expr]
    ) -> Expr:
        """
        Reconstruct a linear MBA in the conjunction basis.

        The basis is ``1``, every single variable, every pairwise conjunction,
        and so on. Because the Boolean assignment order makes each conjunction's
        first nonzero row unique, coefficients can be read from the residual
        vector without general-purpose linear algebra.
        """
        residual = [value & self.mask for value in signature]
        terms: list[Expr] = []

        # Row zero is the all-zero assignment, so it is the constant term. Remove
        # it from all other rows before solving the remaining conjunction terms.
        constant = residual[0] & self.mask
        if constant:
            terms.append(self._const(constant))
        for index in range(1, len(residual)):
            residual[index] = (residual[index] - constant) & self.mask

        for degree in range(1, len(variables) + 1):
            for variable_indexes in combinations(range(len(variables)), degree):
                # The first row where x_i1 & ... & x_ik is true has exactly
                # those assignment bits set. After smaller-degree terms have
                # been subtracted, that row contains this conjunction's
                # coefficient.
                row_index = sum(1 << index for index in variable_indexes)
                coefficient = residual[row_index] & self.mask
                if coefficient == 0:
                    continue

                conjunction = self._conjunction(
                    [variables[index] for index in variable_indexes]
                )
                terms.append(self._multiply(coefficient, conjunction))

                # Subtract the found coefficient from every row where this
                # conjunction is true, so later higher-degree conjunctions see
                # only their still-unexplained residual.
                for assignment in range(len(residual)):
                    if assignment == row_index:
                        continue
                    if all((assignment >> index) & 1 for index in variable_indexes):
                        residual[assignment] = (
                            residual[assignment] - coefficient
                        ) & self.mask

        return self._sum(terms)

    def _refine(
        self, signature: list[int], variables: list[Expr], generic: Expr
    ) -> Expr:
        """
        Try small-variable refinements from the SiMBA paper.

        The conjunction-basis expression is always valid but not always compact.
        For at most three variables, truth tables are small enough to map simple
        Boolean predicates back to bitwise expressions and combine them with the
        observed output coefficients.
        """
        term_count = self._term_count(generic)
        if term_count <= 1:
            return generic

        result_values = set(signature)
        if len(result_values) == 2:
            if signature[0] == 0:
                # One nonzero region: turn that region into a bitwise predicate
                # and multiply it by the nonzero output value.
                return self._expression_for_each_unique_value(signature, variables)

            # Two nonzero values can sometimes be represented as a single
            # negated predicate using ~p == -p - 1 in modular arithmetic.
            negated = self._try_find_negated_single_expression(signature, variables)
            if negated is not None:
                return negated

        if term_count <= 2:
            return generic

        constant = signature[0] & self.mask
        # Many refinement cases are easier after peeling off the constant row.
        # This transforms "constant plus one predicate" into the same shape as
        # the zero-constant cases above.
        shifted = [((value - constant) & self.mask) for value in signature]
        shifted_values = set(shifted)

        if len(shifted_values) == 2:
            return self._sum(
                [
                    self._const(constant),
                    self._expression_for_each_unique_value(shifted, variables),
                ]
            )

        if len(shifted_values) == 3 and constant == 0:
            return self._expression_for_each_unique_value(shifted, variables)

        unique_nonzero = sorted(value for value in shifted_values if value != 0)
        if len(shifted_values) == 4 and constant == 0:
            # If one observed value is the modular sum of two others, we can
            # merge predicate regions and use fewer bitwise terms.
            eliminated = self._try_eliminate_unique_value(
                unique_nonzero, shifted, variables
            )
            if eliminated is not None:
                return eliminated

        if term_count == 3:
            return generic

        if constant == 0:
            return self._expression_for_each_unique_value(shifted, variables)

        eliminated = self._try_eliminate_unique_value(
            unique_nonzero, shifted, variables
        )
        if eliminated is not None:
            return self._sum([self._const(constant), eliminated])

        return generic

    def _try_find_negated_single_expression(
        self, signature: list[int], variables: list[Expr]
    ) -> Expr | None:
        """
        Detect the two-value pattern that can be represented as coeff * ~p.

        If the values are ``a`` and ``2a`` modulo 2^n and the all-zero row is
        ``a``, then rows with ``2a`` form predicate ``p`` and the expression can
        be written as ``(-a) * ~p``.
        """
        values = list(set(signature))
        if len(values) != 2:
            return None

        first, second = values
        if self._is_double_modulo(first, second):
            low, high = second, first
        elif self._is_double_modulo(second, first):
            low, high = first, second
        else:
            return None

        if signature[0] == high:
            return None

        predicate = [int(value == high) for value in signature]
        bitwise = self._lookup_bitwise_expression(predicate, variables)
        if bitwise is None:
            return None
        return self._multiply((-low) & self.mask, self._invert(bitwise))

    def _try_eliminate_unique_value(
        self, values: list[int], signature: list[int], variables: list[Expr]
    ) -> Expr | None:
        """
        Reduce term count by merging regions whose coefficients add modulo 2^n.

        This implements the paper's small truth-table refinement: if value c is
        the modular sum of values a and b, regions carrying c can be included in
        both the a-predicate and b-predicate instead of requiring a third term.
        """
        if len(values) > 4:
            return None

        for i, first in enumerate(values[:-1]):
            for j, second in enumerate(values[i + 1 :], start=i + 1):
                for k, combined in enumerate(values):
                    if k in {i, j}:
                        continue
                    if not self._is_sum_modulo(first, second, combined):
                        continue

                    terms = [
                        self._term_for_value(signature, variables, first, combined),
                        self._term_for_value(signature, variables, second, combined),
                    ]
                    for value in values:
                        if value not in {first, second, combined}:
                            terms.append(
                                self._term_for_value(signature, variables, value)
                            )
                    return self._sum(terms)

        if len(values) < 4:
            return None

        total = sum(values) & self.mask
        for index, value in enumerate(values):
            if (2 * value) & self.mask != total:
                continue
            terms = [
                self._term_for_value(signature, variables, other)
                for other_index, other in enumerate(values)
                if other_index != index
            ]
            return self._sum(terms)

        return None

    def _expression_for_each_unique_value(
        self, signature: list[int], variables: list[Expr]
    ) -> Expr:
        """Build one coefficient * predicate term for each nonzero signature value."""
        terms = [
            self._term_for_value(signature, variables, value)
            for value in sorted(set(signature))
            if value != 0
        ]
        return self._sum(terms)

    def _term_for_value(
        self,
        signature: list[int],
        variables: list[Expr],
        value: int,
        alternate_value: int | None = None,
    ) -> Expr:
        # A predicate row is true wherever the signature equals ``value``. When
        # ``alternate_value`` is supplied, those rows are included too; this is
        # how value-elimination shares one region between two coefficient terms.
        predicate = [
            int(
                current == value
                or (alternate_value is not None and current == alternate_value)
            )
            for current in signature
        ]
        bitwise = self._lookup_bitwise_expression(predicate, variables)
        if bitwise is None:
            raise ValueError("could not build bitwise expression")
        return self._multiply(value, bitwise)

    def _lookup_bitwise_expression(
        self, predicate: list[int], variables: list[Expr]
    ) -> Expr | None:
        """
        Convert a Boolean truth table to a compact bitwise expression.

        Upstream SiMBA ships lookup tables for up to three variables. To avoid a
        bundled table, this implementation recognizes the common compact forms
        directly, then uses Quine-McCluskey minimisation for the rest. A DNF
        fallback follows QM, but QM already covers every table except the
        all-ones table (for which DNF also returns None), so the DNF minterm
        construction is effectively unreachable in practice; it is kept as a
        defensive last resort. Returns None when no bitwise form is produced.
        """
        table = self._table_to_int(predicate)
        variable_tables = [
            self._table_to_int(
                [(assignment >> index) & 1 for assignment in range(1 << len(variables))]
            )
            for index in range(len(variables))
        ]

        if table == 0:
            return self._const(0)

        # A precomputed table gives the globally minimal bitwise formula over
        # {var, ~, &, |, ^} (XOR-aware), recovering compact forms like
        # (d^e)|(d^f) the Quine-McCluskey DNF path below cannot. The predicate is
        # first projected onto the variables it actually depends on: a region
        # like ``~x&(y^z)`` inside a 5-variable expression depends on only three
        # variables, so we look it up over that support and instantiate it over
        # those variables. None means not bitwise-expressible (all-ones); we then
        # fall through to the legacy path (which also bails).
        support = self._predicate_support(predicate, len(variables))
        if len(support) <= _MAX_TABLE_VARS:
            projected = self._project_predicate(predicate, support, len(variables))
            recipe = minimal_bitwise_recipe(projected, len(support))
            if recipe is not None:
                return instantiate_recipe(
                    recipe,
                    [variables[index] for index in support],
                    self._invert,
                    self._conjunction,
                    self._or,
                    self._xor,
                )

        for index, variable_table in enumerate(variable_tables):
            if table == variable_table:
                return variables[index]

        for degree in range(2, len(variables) + 1):
            for indexes in combinations(range(len(variables)), degree):
                # Check simple n-ary XOR/AND/OR over every variable subset before
                # falling back to DNF. These are the forms that make examples
                # like x ^ y and x | y stay compact.
                xor_table = 0
                and_table = (1 << (1 << len(variables))) - 1
                or_table = 0
                for index in indexes:
                    xor_table ^= variable_tables[index]
                    and_table &= variable_tables[index]
                    or_table |= variable_tables[index]
                selected = [variables[index] for index in indexes]
                if table == xor_table:
                    return self._xor(selected)
                if table == and_table:
                    return self._conjunction(selected)
                if table == or_table:
                    return self._or(selected)

        # Try Quine-McCluskey minimisation before falling back to DNF.
        qm_terms = _qm_minimise(table, len(variables))
        qm_expr = self._build_qm_bitwise(qm_terms, variables)
        if qm_expr is not None:
            # Return the per-region QM bitwise expression unrefined: algebraic
            # polishing is applied once to the fully-assembled reconstruction at
            # the end of :meth:`simplify` (via :func:`_bitwise_refine`), not per
            # region — running the §5.2 rules per region perturbs SimBA's
            # multi-coefficient assembly, whereas refining the complete output
            # is sound (see the comment at the ``simplify`` return site).
            return qm_expr
        # Defensive last resort: DNF. In practice QM above succeeds for every
        # table except the all-ones table, and for that table _dnf_expression
        # also returns None (its row-0 minterm can't be built), so this call
        # currently only ever returns None. Kept in case QM coverage changes.
        return self._dnf_expression(predicate, variables)

    def _canon(self, expr: Expr):
        """A commutativity-insensitive canonical key for structural matching."""
        if isinstance(expr, ExprOp):
            args = [self._canon(arg) for arg in expr.args]
            if expr.op in ("+", "*", "&", "|", "^"):
                args = sorted(args, key=repr)
            return (expr.op, tuple(args))
        return ("leaf", repr(expr))

    def _collapse_product_identities(self, expr: Expr) -> Expr:
        """Bottom-up rewrite of ``(A&B)·(A|B)+(A&~B)·(~A&B)`` sub-sums to ``A·B``.

        Returns ``expr`` unchanged (same object) when nothing matched, so callers
        can cheaply detect a no-op.
        """
        if not isinstance(expr, ExprOp):
            return expr
        new_args = [self._collapse_product_identities(arg) for arg in expr.args]
        changed = any(a is not b for a, b in zip(new_args, expr.args))
        current = ExprOp(expr.op, *new_args) if changed else expr
        if current.op == "+":
            collapsed = self._collapse_sum_products(current)
            if collapsed is not None:
                return collapsed
        return current

    def _collapse_sum_products(self, sum_expr: ExprOp) -> Expr | None:
        """Replace identity-matching product pairs inside one ``+`` node."""
        terms = list(sum_expr.args)
        any_change = False
        progress = True
        while progress:
            progress = False
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    product = self._match_product_identity(terms[i], terms[j])
                    if product is None:
                        continue
                    a, b = product
                    if not self._validates_local(terms[i], terms[j], ExprOp("*", a, b)):
                        continue
                    # Simplify the product's factors: the obfuscation often hides
                    # a linear MBA inside one factor (``B·y`` where ``B`` reduces
                    # to ``a``), so reduce each before forming ``A·B``. Both are
                    # equivalent to the original factors, so the validated
                    # identity ``terms[i]+terms[j] == A·B`` still holds.
                    a = _SimbaSimplifier(a, depth=self._depth + 1).simplify()
                    b = _SimbaSimplifier(b, depth=self._depth + 1).simplify()
                    folded = ExprOp("*", a, b)
                    terms = [
                        term for k, term in enumerate(terms) if k not in (i, j)
                    ] + [folded]
                    any_change = True
                    progress = True
                    break
                if progress:
                    break
        if not any_change:
            return None
        if len(terms) == 1:
            return terms[0]
        return ExprOp("+", *terms)

    def _match_product_identity(self, t1: Expr, t2: Expr):
        """Return ``(A, B)`` if ``{t1, t2}`` look like ``(A&B)·(A|B)``/``(A&~B)·(~A&B)``.

        ``A`` and ``B`` are read off the clean ``(A&B)·(A|B)`` factor; the other
        term only needs to be a product of two non-constants. The numeric
        identity ``t1 + t2 == A·B`` is then confirmed by the caller's
        :meth:`_validates_local` guard, which is robust to how the second term's
        ``~`` operands happen to be normalised (``~~(y-1)`` vs ``y-1``).
        """
        for first, second in ((t1, t2), (t2, t1)):
            extracted = self._as_and_or_product(first)
            if extracted is None:
                continue
            if (
                isinstance(second, ExprOp)
                and second.op == "*"
                and sum(1 for arg in second.args if not isinstance(arg, ExprInt)) >= 2
            ):
                return extracted
        return None

    def _as_and_or_product(self, term: Expr):
        """If ``term == (A&B)·(A|B)`` return ``(A, B)``, else None."""
        if not (isinstance(term, ExprOp) and term.op == "*" and len(term.args) == 2):
            return None
        for conj, disj in (term.args, term.args[::-1]):
            if not (
                isinstance(conj, ExprOp)
                and conj.op == "&"
                and len(conj.args) == 2
                and isinstance(disj, ExprOp)
                and disj.op == "|"
                and len(disj.args) == 2
            ):
                continue
            a, b = conj.args
            expected = ExprOp("|", a, b)
            if self._canon(disj) == self._canon(expected):
                return (a, b)
        return None

    def _is_andnot_product(self, term: Expr, a: Expr, b: Expr) -> bool:
        """True iff ``term == (A&~B)·(~A&B)`` for the given ``A``, ``B``."""
        if not (isinstance(term, ExprOp) and term.op == "*" and len(term.args) == 2):
            return False
        wanted = {
            self._canon(self._conjunction([a, self._invert(b)])),
            self._canon(self._conjunction([self._invert(a), b])),
        }
        actual = {self._canon(term.args[0]), self._canon(term.args[1])}
        return wanted == actual

    def _validates_local(self, t1: Expr, t2: Expr, folded: Expr) -> bool:
        """Confirm ``t1 + t2 == folded`` on random inputs (soundness guard)."""
        terminals = sorted(
            _collect_terminals(t1)
            | _collect_terminals(t2)
            | _collect_terminals(folded),
            key=repr,
        )
        rng = _DeterministicRandom(0x5119BA)
        for _ in range(48):
            env = {term: rng.next64() for term in terminals}
            try:
                left = (
                    _eval_expr_int(t1, env, self.mask)
                    + _eval_expr_int(t2, env, self.mask)
                ) & self.mask
                right = _eval_expr_int(folded, env, self.mask)
            except (KeyError, ValueError):
                return False
            if left != right:
                return False
        return True

    def _predicate_support(self, predicate: list[int], n_vars: int) -> list[int]:
        """Return the variables the predicate actually depends on.

        A variable is irrelevant when flipping its bit never changes the
        predicate value; projecting onto the remaining variables lets a region
        that uses only a few of many variables (``~x&(y^z)`` inside a 5-variable
        expression) reuse the minimal ≤3-variable bitwise table.
        """
        support: list[int] = []
        for index in range(n_vars):
            bit = 1 << index
            for assignment in range(1 << n_vars):
                if assignment & bit:
                    continue
                if predicate[assignment] != predicate[assignment | bit]:
                    support.append(index)
                    break
        return support

    def _project_predicate(
        self, predicate: list[int], support: list[int], n_vars: int
    ) -> int:
        """Pack the predicate's truth table over its ``support`` variables.

        The predicate is independent of the variables outside ``support`` (by
        construction in :meth:`_predicate_support`), so reading off one
        consistent full assignment per sub-assignment is well defined.
        """
        table = 0
        for sub in range(1 << len(support)):
            assignment = 0
            for position, index in enumerate(support):
                if (sub >> position) & 1:
                    assignment |= 1 << index
            if predicate[assignment]:
                table |= 1 << sub
        return table

    def _dnf_expression(
        self, predicate: list[int], variables: list[Expr]
    ) -> Expr | None:
        """
        Build a disjunctive-normal-form predicate.

        DNF is only a defensive fallback (see :meth:`_lookup_bitwise_expression`:
        Quine-McCluskey already covers every reachable table, so this is not
        exercised in practice). If the predicate includes the all-zero row
        (assignment 0 enabled), building that minterm would require a constant
        true expression; returning None lets the caller abandon that refinement.
        """
        terms = []
        for assignment, enabled in enumerate(predicate):
            if not enabled:
                continue
            if assignment == 0:
                return None
            literals = [
                variables[index]
                if (assignment >> index) & 1
                else self._invert(variables[index])
                for index in range(len(variables))
            ]
            terms.append(self._conjunction(literals))
        return self._or(terms) if terms else self._const(0)

    def _simplify_fewer_variables(self, expr: Expr) -> Expr:
        """Rerun SiMBA after generic reconstruction removes unused variables."""
        occurring = _collect_atoms(expr)
        if len(occurring) > 3:
            return expr
        inner = _SimbaSimplifier(expr, depth=self._depth + 1)
        return inner.simplify()

    def _effective_variable_count(self, expr: Expr) -> int:
        return len(_collect_atoms(expr))

    def _table_to_int(self, values: list[int]) -> int:
        """Pack a truth table into an integer so tables can be compared cheaply."""
        result = 0
        for index, value in enumerate(values):
            if value:
                result |= 1 << index
        return result

    def _build_qm_bitwise(
        self, prime_implicants: list[tuple[int, int]], variables: list[Expr]
    ) -> Expr | None:
        """
        Convert Quine-McCluskey prime implicants to a miasm bitwise expression.

        Each implicant is (value, mask) where bits in ``mask`` are don't-cares
        and bits in ``value`` outside the mask are the required literal values.
        A fully-masked implicant (all literals don't-care) means constant 1,
        which we cannot express in the bitwise basis — return None so the
        caller can fall back to DNF (which also bails on the row-0=1 case).
        """
        if not prime_implicants:
            return self._const(0)
        full_mask = (1 << len(variables)) - 1
        terms: list[Expr] = []
        for value, mask in prime_implicants:
            if mask == full_mask:
                # constant 1 implicant — not expressible as a bitwise term.
                return None
            literals = []
            for index in range(len(variables)):
                if (mask >> index) & 1:
                    continue
                if (value >> index) & 1:
                    literals.append(variables[index])
                else:
                    literals.append(self._invert(variables[index]))
            terms.append(self._conjunction(literals))
        return self._or(terms)

    def _is_sum_modulo(self, first: int, second: int, result: int) -> bool:
        return (first + second - result) % self.modulus == 0

    def _is_double_modulo(self, result: int, value: int) -> bool:
        return (2 * value - result) % self.modulus == 0

    def _term_count(self, expr: Expr) -> int:
        if isinstance(expr, ExprOp) and expr.op == "+":
            return sum(self._term_count(arg) for arg in expr.args)
        return 1

    def _const(self, value: int) -> ExprInt:
        return ExprInt(value & self.mask, self.size)

    def _multiply(self, coefficient: int, expr: Expr) -> Expr:
        """Build coefficient * expr while preserving SiMBA's modular constants."""
        coefficient &= self.mask
        if coefficient == 0:
            return self._const(0)
        if coefficient == 1:
            return expr
        if isinstance(expr, ExprInt):
            return self._const(coefficient * int(expr))
        if coefficient == self.mask:
            # -1 · expr — a unary negation is one node smaller than ``mask·expr``
            # and is what the reference forms use (``-x`` rather than ``0xFF..F·x``).
            return ExprOp("-", expr)
        return ExprOp("*", self._const(coefficient), expr)

    def _sum(self, terms: list[Expr]) -> Expr:
        """Build a variadic sum, dropping zeros and using subtraction for
        negated terms.

        Negated terms (unary ``-X``) are collected and emitted with a binary
        subtraction, so ``x + -y`` is built as ``x - y`` (one node smaller and the
        canonical reference shape).
        """
        filtered = [
            term for term in terms if not (isinstance(term, ExprInt) and int(term) == 0)
        ]
        if not filtered:
            return self._const(0)
        if len(filtered) == 1:
            return filtered[0]

        def is_negated(term: Expr) -> bool:
            return isinstance(term, ExprOp) and term.op == "-" and len(term.args) == 1

        positive = [term for term in filtered if not is_negated(term)]
        negative = [term.args[0] for term in filtered if is_negated(term)]
        if not negative:
            return ExprOp("+", *filtered)
        # ``mask - X == ~X`` exactly — emit the canonical bitwise-not form.
        if (
            len(positive) == 1
            and len(negative) == 1
            and isinstance(positive[0], ExprInt)
            and int(positive[0]) == self.mask
        ):
            return self._invert(negative[0])
        if not positive:
            inner = negative[0] if len(negative) == 1 else ExprOp("+", *negative)
            return ExprOp("-", inner)
        positive_expr = positive[0] if len(positive) == 1 else ExprOp("+", *positive)
        return ExprOp("-", positive_expr, *negative)

    def _conjunction(self, terms: list[Expr]) -> Expr:
        if not terms:
            return self._const(self.mask)
        if len(terms) == 1:
            return terms[0]
        return ExprOp("&", *terms)

    def _or(self, terms: list[Expr]) -> Expr:
        if not terms:
            return self._const(0)
        if len(terms) == 1:
            return terms[0]
        return ExprOp("|", *terms)

    def _xor(self, terms: list[Expr]) -> Expr:
        if not terms:
            return self._const(0)
        if len(terms) == 1:
            return terms[0]
        return ExprOp("^", *terms)

    def _invert(self, expr: Expr) -> Expr:
        if isinstance(expr, ExprInt):
            return self._const(~int(expr))
        return ExprOp("^", expr, self._const(self.mask))
