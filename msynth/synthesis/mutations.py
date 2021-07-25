from __future__ import annotations

from random import choice, getrandbits
from typing import List

from miasm.expression.expression import Expr, ExprInt
from msynth.synthesis.grammar import Grammar
from msynth.synthesis.state import SynthesisState
from msynth.utils.expr_utils import get_subexpressions


class Mutator:
    """
    Mutator to perform mutations on a synthesis state.

    The mutator interacts with the grammar to perform
    minor modifications to the synthesis state. It
    provides the following mutations:

    - replace subexpression with leaf
    - replace subexpression with expression
    - downcast expression

    The first two mutations are enabled by default. The mutator
    provides the functionality to enable additional mutations
    by analyzing a provided expression that represents a function
    f(x0, ..., xi). 

    Mutations operate on the synthesis state by modifying its `expr_ast`
    as well as updating it replacement dictionary accordingly. At the end of
    each mutation, the synthesis state is cleaned up to avoid dead mutations.

    Enabled mutations are managed in the list `mutations`. The list `sizes_casting`
    stores sizes used in downcasting and upcasting mutations.

    Attributes:
        grammar (Grammar): Grammar to generate expressions.
        sizes_casting (List(int)): Sizes for down/upcasting mutations.
        mutations (List(Any)): List of active mutations.
    """

    def __init__(self, grammar: Grammar):
        """
        Initializes a Mutator instance.

        Args:
            grammar (Grammar): Grammar to generate expressions
        """
        self.grammar = grammar
        self.sizes_casting: List[int] = []
        self.mutations = [
            self.replace_subexpression_with_leaf,
            self.replace_subexpression_with_expression
        ]

    def gen_from_expression(expr: Expr, grammar: Grammar) -> Mutator:
        """
        Generates a mutator for a provided expression that 
        represents a function f(x0, ..., xi). It enables
        different mutations based on the provided expression

        Args:
            expr (Expr): Expression in Miasm IR.
            grammar (Grammar): Grammar to generate expressions.

        Returns:
            Mutator: Mutator customized for provided expression.
        """
        # init mutator
        mutator = Mutator(grammar)

        # enable type casting mutations if necessary
        mutator.maybe_enable_casting(expr)

        return mutator

    def maybe_enable_casting(self, expr: Expr) -> None:
        """
        Checks if type cast mutations should be enabled.

        The mutations will be enabled if the provided expression 
        contains subexpressions of different sizes. For each size,
        the bitmask will be calculated and used as size for type castings.

        Args:
            expr ([type]):  Expression in Miasm IR.
        """
        # calculate bitmasks
        bitmasks = list(
            sorted(set([(1 << e.size) - 1 for e in get_subexpressions(expr)])))

        # more than one size
        if len(bitmasks) > 1:
            # assign sizes
            self.sizes_casting = bitmasks
            # add downcasting mutation
            self.mutations.append(self.downcast_expression)

    def mutate(self, state: SynthesisState) -> SynthesisState:
        """
        Applies different mutations to a synthesis state.

        It randomly samples a value n and performs n random 
        enabled mutations sequentially to the synthesis state.

        Args:
            state (SynthesisState): XXX

        Returns:
            SynthesisState: XXX
        """
        for _ in range(getrandbits(32) % 5 + 1):
            state = choice(self.mutations)(state)
        return state

    def replace_subexpression_with_leaf(self, state: SynthesisState) -> SynthesisState:
        """
        Mutation to replace a subexpression with a leaf.

        The mutation randomly selects a subexpression and a random
        terminal expression. Then, it replaces the variable in the state 
        accordingly. Finally, the state in cleaned up.

        Example:

        The random subexpression x + y from (x + y) + z is replaced with
        a variable x, leading to x + z.

        Args:
            state (SynthesisState): State to mutate.

        Returns:
            SynthesisState: Mutated state.
        """
        # choose random subexpression from AST
        sub_expr = choice(get_subexpressions(state.expr_ast))
        # generate new fresh variable
        v = self.grammar.gen_fresh_var_of_size(sub_expr.size)

        # replace subexpression with fresh variable in AST
        state.expr_ast = state.expr_ast.replace_expr({sub_expr: v})
        # update replacement dictionary
        state.replacements.update(
            {v: self.grammar.get_rand_var_of_size(v.size)})

        # clean state
        state.cleanup()

        return state

    def replace_subexpression_with_expression(self, state: SynthesisState) -> SynthesisState:
        """
        Mutation to replace a subexpression with an expression.

        The mutation randomly selects a subexpression and generates a new
        random expression from the grammar. Then, it replaces the subexpression in the state 
        accordingly. Finally, the state in cleaned up.

        The newly generated expression is an AST of depth 1.

        Example:

        The random subexpression x + y from (x + y) + z is replaced with
        an AST of depth 1, x * x, leading to (x * x) + z.

        Args:
            state (SynthesisState): State to mutate.

        Returns:
            SynthesisState: Mutated state.
        """
        # choose random expression from AST
        sub_expr = choice(get_subexpressions(state.expr_ast))
        # generate new expression/AST and replacement dictionary
        expr, repl = self.grammar.gen_expr_for_state()

        # if expression size and subexpression size do not match, return
        # TODO: maybe perform typecasting here
        if sub_expr.size != expr.size:
            return state

        # replace expression in AST
        state.expr_ast = state.expr_ast.replace_expr({sub_expr: expr})
        # update replacement dictionary
        state.replacements.update(repl)

        # clean state
        state.cleanup()

        return state

    def downcast_expression(self, state: SynthesisState) -> SynthesisState:
        """
        Mutation to downcast a subexpression.

        The mutation randomly selects a subexpression and chooses a size that is smaller than
        the subexpression's size. Afterward, it semantically downcasts the expression by applying
        it's bit mask and updates the state accordingly. Finally, the state in cleaned up.

        Example:

        The random subexpression x + y from (x + y) + z is downcasted to a byte
        value, leading to ((x + y) & 0xff) + z.

        Args:
            state (SynthesisState): State to mutate.

        Returns:
            SynthesisState: Mutated state.
        """
        # choose random expression from AST
        sub_expr = choice(get_subexpressions(state.expr_ast))

        # choose a random size for downcasting
        value = choice(self.sizes_casting)
        # repeat until the chosen size is smaller than the chosen subexpressions'
        while sub_expr.size - 1 > value:
            value = choice(self.sizes_casting)

        # downcast the subexpression
        repl = sub_expr & ExprInt(value, sub_expr.size)
        # replace expression in AST
        state.expr_ast = state.expr_ast.replace_expr({sub_expr: repl})

        # clean state
        state.cleanup()

        return state
