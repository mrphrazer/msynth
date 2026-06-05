"""
Precomputed minimal bitwise-formula table for SiMBA's per-region lookup.

For a boolean predicate over ``n <= _MAX_TABLE_VARS`` cube variables (each taking
value 0 or 1), this module returns the *globally minimal* (by Miasm graph node
count) bitwise expression over the grammar ``{var, ~, &, |, ^}`` whose value on
the Boolean cube equals that predicate.

Why this exists
---------------
SiMBA's reconstruction expresses a linear MBA as ``Σ c_k · f_k`` where each
``f_k`` is a 0/1-valued boolean function of the atoms. The old per-region lookup
(:meth:`_SimbaSimplifier._lookup_bitwise_expression`) used Quine-McCluskey, which
only ever produces sum-of-products (DNF) forms — it never discovers XOR-aware
compactions like ``(d^e)|(d^f)`` or ``~x&(y^z)``, which is exactly what the SiMBA
reference's bundled lookup tables provide. This module reproduces those tables by
brute-force Dijkstra over the expression grammar, ranked by the real node-count
metric, so each region is reconstructed minimally.

Cube / soundness invariant
--------------------------
Variables are evaluated as full-width bitvectors restricted to ``{0, 1}`` and
``~`` is ``x ^ mask`` (full width). A bare ``~x`` therefore evaluates to ``mask``
(not 1), so only expressions whose cube value is in ``{0, 1}`` on *every*
assignment are valid 0/1 predicate representations; the generator keeps only
those as table entries (other expressions are retained solely as building blocks).
Predicates are always false at the all-zero row (SiMBA peels the constant first),
so every table entry has cube row 0 equal to 0; such tables are always
representable. The all-ones table is *not* bitwise-expressible and maps to
``None`` (matching the old contract).

The recipe is stored as a structural tree of variable indexes and is instantiated
through the caller's width-aware builders (``_invert``/``_conjunction``/``_or``/
``_xor``) so all-ones constants and bit width stay consistent with the rest of
SiMBA's output.
"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, Optional, Tuple

# Recipe = structural bitwise tree over variable indexes:
#   ("var", i)
#   ("not", child)
#   ("and"|"or"|"xor", child_a, child_b)
Recipe = tuple

_MAX_TABLE_VARS = 3
# Cost ceiling for the Dijkstra search. The hardest 3-variable boolean function
# has a minimal formula well under this; raising it only widens the (already
# complete) search. Generation is one-time and cached.
_MAX_COST = 16
_GEN_SIZE = 8  # bit width used for generation; node counts are width-independent
_GEN_MASK = (1 << _GEN_SIZE) - 1

# Cap on how many equal-cost (tied) recipes to retain per truth table. Multiple
# minimal forms let the caller pick the one that shares the most subexpressions
# with sibling terms (``~x&(y^z)`` vs ``(x|y)^(x|z)`` are both minimal, but the
# former shares ``y^z`` with a neighbouring ``x&(y^z)`` term).
_MAX_TIED_RECIPES = 8

# table[n] : dict mapping the 0/1 truth-table integer -> minimal Recipe
_TABLES: Dict[int, Dict[int, Recipe]] = {}
_COSTS: Dict[int, Dict[int, int]] = {}
# all_recipes[n] : dict mapping truth table -> list of tied minimal-cost recipes
_ALL_RECIPES: Dict[int, Dict[int, list]] = {}


def _eval_recipe(recipe: Recipe, assignment: int) -> int:
    """Evaluate a recipe on one cube assignment with full-width ``~`` semantics."""
    kind = recipe[0]
    if kind == "var":
        return (assignment >> recipe[1]) & 1
    if kind == "not":
        return _eval_recipe(recipe[1], assignment) ^ _GEN_MASK
    a = _eval_recipe(recipe[1], assignment)
    b = _eval_recipe(recipe[2], assignment)
    if kind == "and":
        return a & b
    if kind == "or":
        return a | b
    if kind == "xor":
        return a ^ b
    raise ValueError(f"bad recipe kind {kind!r}")


def _value_vector(recipe: Recipe, n: int) -> Tuple[int, ...]:
    return tuple(_eval_recipe(recipe, a) for a in range(1 << n))


def _recipe_cost(recipe: Recipe) -> int:
    """Node count matching ``len(expr.graph().nodes())``: unique sub-nodes, with
    ``~`` costed as ``x ^ mask`` (an extra xor node plus the shared mask const)."""
    nodes: set = set()
    has_not = [False]

    def walk(r: Recipe) -> Tuple:
        kind = r[0]
        if kind == "var":
            key = ("var", r[1])
            nodes.add(key)
            return key
        if kind == "not":
            child = walk(r[1])
            has_not[0] = True
            key = ("xor", child, ("int", _GEN_MASK))
            nodes.add(key)
            return key
        a = walk(r[1])
        b = walk(r[2])
        key = (kind, a, b)
        nodes.add(key)
        return key

    walk(recipe)
    if has_not[0]:
        nodes.add(("int", _GEN_MASK))
    return len(nodes)


def _truth_table_int(vector: Tuple[int, ...]) -> Optional[int]:
    """Return the packed 0/1 truth table, or None when any row is not 0/1."""
    table = 0
    for index, value in enumerate(vector):
        if value == 0:
            continue
        if value == 1:
            table |= 1 << index
        else:
            return None
    return table


def _build_table(n: int) -> None:
    """Dijkstra over the bitwise grammar; record min-cost recipe per value vector."""
    # best[value_vector] = (cost, recipe). Variables seed the frontier.
    best: Dict[Tuple[int, ...], Tuple[int, Recipe]] = {}
    table: Dict[int, Recipe] = {}
    costs: Dict[int, int] = {}
    all_recipes: Dict[int, list] = {}
    heap: list = []
    counter = 0

    def consider(recipe: Recipe) -> None:
        nonlocal counter
        vector = _value_vector(recipe, n)
        cost = _recipe_cost(recipe)
        if cost > _MAX_COST:
            return
        prev = best.get(vector)
        # Even when this vector already has an equal-cost recipe, keep exploring
        # so the *truth table* (a different key) can collect tied alternatives.
        if prev is None or prev[0] > cost:
            best[vector] = (cost, recipe)
            heapq.heappush(heap, (cost, counter, recipe, vector))
            counter += 1
        elif prev[0] < cost:
            pass  # strictly worse for this vector — but may still tie a table
        tt = _truth_table_int(vector)
        if tt is not None:
            cur = costs.get(tt)
            if cur is None or cost < cur:
                costs[tt] = cost
                table[tt] = recipe
                all_recipes[tt] = [recipe]
            elif cost == cur and len(all_recipes[tt]) < _MAX_TIED_RECIPES:
                if recipe not in all_recipes[tt]:
                    all_recipes[tt].append(recipe)

    for index in range(n):
        consider(("var", index))

    while heap:
        cost, _, recipe, vector = heapq.heappop(heap)
        if best.get(vector, (None,))[0] != cost or best[vector][1] != recipe:
            # superseded by a cheaper recipe for the same vector
            continue
        if cost >= _MAX_COST:
            continue
        # Unary negation.
        consider(("not", recipe))
        # Binary combinations with every settled recipe.
        for other_cost, other_recipe in list(best.values()):
            if other_cost + cost + 1 > _MAX_COST:
                continue
            consider(("and", recipe, other_recipe))
            consider(("or", recipe, other_recipe))
            consider(("xor", recipe, other_recipe))

    _TABLES[n] = table
    _COSTS[n] = costs
    _ALL_RECIPES[n] = all_recipes


def _ensure(n: int) -> None:
    if n not in _TABLES:
        _build_table(n)


def minimal_bitwise_recipe(table: int, n: int) -> Optional[Recipe]:
    """Minimal-node recipe for the ``n``-var boolean function ``table``.

    ``table`` is the packed truth table (bit ``i`` = function value at cube
    assignment ``i``). Returns ``None`` when ``n > _MAX_TABLE_VARS`` or the table
    is not representable in the bitwise basis (only the all-ones table, given the
    row-0 = 0 usage contract).
    """
    if n <= 0 or n > _MAX_TABLE_VARS:
        return None
    _ensure(n)
    return _TABLES[n].get(table)


def minimal_bitwise_recipes(table: int, n: int) -> list:
    """All retained minimal-cost recipes for the ``n``-var function ``table``.

    Returns an empty list when the table is unsupported / not expressible. The
    caller uses the alternatives to maximise subexpression sharing across the
    terms of a decomposition.
    """
    if n <= 0 or n > _MAX_TABLE_VARS:
        return []
    _ensure(n)
    return list(_ALL_RECIPES[n].get(table, []))


def recipe_node_cost(table: int, n: int) -> Optional[int]:
    if n <= 0 or n > _MAX_TABLE_VARS:
        return None
    _ensure(n)
    return _COSTS[n].get(table)


def instantiate_recipe(
    recipe: Recipe,
    variables,
    invert: Callable,
    conjunction: Callable,
    disjunction: Callable,
    xor: Callable,
):
    """Instantiate a recipe into a Miasm expression via the caller's builders."""
    kind = recipe[0]
    if kind == "var":
        return variables[recipe[1]]
    if kind == "not":
        return invert(
            instantiate_recipe(recipe[1], variables, invert, conjunction, disjunction, xor)
        )
    a = instantiate_recipe(recipe[1], variables, invert, conjunction, disjunction, xor)
    b = instantiate_recipe(recipe[2], variables, invert, conjunction, disjunction, xor)
    if kind == "and":
        return conjunction([a, b])
    if kind == "or":
        return disjunction([a, b])
    if kind == "xor":
        return xor([a, b])
    raise ValueError(f"bad recipe kind {kind!r}")
