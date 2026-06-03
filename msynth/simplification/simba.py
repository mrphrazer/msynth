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
    Post-QM polish on a synthesised bitwise expression.

    Reuses msynth's GAMBA §5.2 no-grow preprocessor (idempotence, De Morgan,
    absorption, redundancy, complement-pair, XOR-collapses, …) to refine the
    Quine-McCluskey output. Corresponds in spirit to upstream GAMBA's
    ``BitwiseFactory.refine`` (XOR-insertion · negation-flipping · common-
    factor extraction) — the unguarded §5.2 rules cover the negation-flip
    and XOR-collapse cases. Common-factor extraction is intentionally NOT
    invoked here: the guarded ``ring_normalize`` / ``factor_common_subterm``
    rules in the post-rewriter widen the cube SimBA reconstructs over,
    producing a refined form that fails the soundness check downstream
    (see ``test_simba_4var_qm_produces_compact_form``). The preprocessor
    is no-grow by construction, so the refined form cannot become larger
    from rule application alone. The net-shrink guard below is belt-and-
    braces against the rare case where bottom-up normalisation rebuilds
    equal-size nodes differently.

    Args:
        expr: A bitwise Expr produced by ``_build_qm_bitwise`` (Quine-
            McCluskey reconstruction) or any equivalent SimBA stage.

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


def _apply_op_rule(
    op: str, args: tuple, arg_kinds: list, parent_size: int
) -> _ExpressionKind | None:
    """
    Per-op linear-MBA classification rule.

    Given an ExprOp's operator string, args, and the classified kinds
    of those args, return the kind of the whole op — or ``None`` if no
    rule in the linear-MBA fragment matches (signalling that the op
    should be treated as an opaque BITWISE atom by the caller).
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
    Slice/Compose/Mem leaves to every node that the strict linear-MBA
    classifier rejects: when no per-op rule in :func:`_apply_op_rule`
    matches, the node is returned as a BITWISE atom. Soundness rests
    on three properties that hold for every miasm pure-function node:

    1. Determinism per cube assignment — when the inner atoms take
       fixed integer values, the node takes a deterministic integer
       value.
    2. Structural dedup — two textually equal occurrences map to the
       same atom via miasm's ``Expr.__hash__`` / ``__eq__``.
    3. Width match — checked here via ``expr.size != parent_size``,
       so ``env[node]`` and the cube modulus align.

    ``(None, True)`` is returned only when width fundamentally
    mismatches — the no-op short-circuit signal used by callers.
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
            # Width fundamentally doesn't fit; can't atomise either.
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


class _SimbaSimplifier:
    def __init__(self, expr: Expr):
        self.expr = expr
        self.size = expr.size
        self.modulus = 1 << self.size
        self.mask = self.modulus - 1
        self._classify_cache: dict[Expr, tuple[_ExpressionKind | None, bool]] = {}
        self.variables = _collect_atoms(expr)

    def simplify(self) -> Expr:
        if self.size <= 0:
            return self.expr

        # Under the atomisation extension, the only thing that can
        # produce a no-op signal is a fundamental width mismatch — the
        # classifier always finds an atom for everything else (at worst,
        # the whole expression becomes a single opaque BITWISE atom and
        # SiMBA's reconstruction returns it unchanged).
        if self._classify(self.expr) is None:
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

        return simplified

    def _classify(self, expr: Expr) -> _ExpressionKind | None:
        """
        Return the linear-MBA kind of ``expr`` or None if width
        fundamentally doesn't fit the cube model.

        Under the atomisation extension, the only non-None failure
        mode is a size mismatch with ``self.size``. Every other node
        — including operators outside the linear-MBA fragment, like
        shifts, rotations, division, multiplication of two non-arith
        operands, and ExprCond — classifies as BITWISE because the
        cube reasoning treats it as an opaque atom. See the module-
        level :func:`_classify` for the soundness sketch.
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

        # First build the always-valid conjunction-basis representation. For up
        # to three variables, try the paper's lookup/refinement rules afterward
        # because those often recover compact bitwise forms such as x ^ y.
        generic = self._generic_linear_combination(signature, variables)
        if len(variables) <= 3:
            return self._refine(signature, variables, generic)
        return generic

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
        directly and falls back to a DNF expression for remaining predicates.
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
            # NOTE: :func:`_bitwise_refine` is wired into the module but
            # NOT called here — running the §5.2 preprocessor on per-region
            # QM output triggers downstream SimBA-reconstruction tests in
            # ``tests/simplification/test_simba_atoms.py`` (the affine-
            # combination / division-atom / three-atom-mix Z3 checks). The
            # rules are individually sound; the interaction with SimBA's
            # multi-coefficient assembly is the open question. Follow-up:
            # narrow refine to a hand-selected rule subset (XOR-collapse,
            # De Morgan only) and re-test, or run refine after the full
            # SimBA output assembly rather than per-region.
            return qm_expr
        # Final fallback: DNF (which may return None for row-0=1 cases)
        return self._dnf_expression(predicate, variables)

    def _dnf_expression(
        self, predicate: list[int], variables: list[Expr]
    ) -> Expr | None:
        """
        Build a disjunctive-normal-form predicate.

        DNF is only a fallback for small truth tables. If the predicate includes
        the all-zero row, building that minterm would require a constant true
        expression; returning None lets the caller abandon that refinement.
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
        inner = _SimbaSimplifier(expr)
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
        return ExprOp("*", self._const(coefficient), expr)

    def _sum(self, terms: list[Expr]) -> Expr:
        """Build a variadic sum, dropping explicit zero terms."""
        filtered = [
            term for term in terms if not (isinstance(term, ExprInt) and int(term) == 0)
        ]
        if not filtered:
            return self._const(0)
        if len(filtered) == 1:
            return filtered[0]
        return ExprOp("+", *filtered)

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
