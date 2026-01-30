from random import choice, getrandbits
from typing import Dict, List, Tuple

from miasm.expression.expression import Expr, ExprId, ExprSlice


class Grammar:
    """
    Grammar generates random expressions in Miasm IL used for synthesis.

    It expects as input a size of a expression representing a function
    f(x0, ..., xi) and a list of variables x0, ... xi. Then, the grammar generates
    random expressions based on such variables. To be compatible with the SynthesisState,
    it uses fresh variables in the random expressions and builds replacement dictionaries
    that randomly map fresh variables to x0 ... xi. The grammar can deal with variables
    of differed sizes.

    It stores a variable index, a size of the expressions to generate
    and a list of variables. To ensure that each expression uses fresh variables,
    it increments the variable index each time a fresh variable has been generated. Fresh
    variables are prefixed with a string.

    Attributes:
        variable_index (int): Index of the latest fresh variable.
        size (int): Size for expressions to generate.
        variables (List[Expr]): List of variables x0 ... xi.
        _var_name_prefix (str): Prefix used for variable names.
    """

    def __init__(self, size: int, variables: List[Expr]):
        """
        Initializes a Grammar instance.

        Args:
            size (int): Size for expressions to generate.
            variables (List[Expr]): List of variables x0 ... xi.
        """
        self.variable_index: int = 0
        self.size: int = size
        self.variables: List[Expr] = variables
        self._var_name_prefix: str = "grammar_var"

    def gen_fresh_var(self) -> Expr:
        """
        Generates a fresh variable of a random size.

        Returns:
            Expr: Variable in Miasm IR.
        """
        self.variable_index += 1
        return ExprId(
            f"{self._var_name_prefix}{self.variable_index}", choice(self.variables).size
        )

    def gen_fresh_var_of_size(self, size: int) -> Expr:
        """
        Generates a fresh variable with a given size.

        Args:
            size (int): Size of variable.

        Returns:
            Expr: Variable in Miasm IL.
        """
        self.variable_index += 1
        return ExprId(f"{self._var_name_prefix}{self.variable_index}", size)

    def get_rand_var_of_size(self, size: int) -> Expr:
        """
        Randomly chooses a variable x0 .... x1 with a given size.

        Args:
            size (int): Size of variable.

        Returns:
            Expr: Variable in Miasm IL.
        """
        # choose random variable
        v = choice(self.variables)
        # repeat until variable size matches given size
        while v.size != size:
            v = choice(self.variables)
        return v

    def gen_expr_for_state(self) -> Tuple[Expr, Dict[Expr, Expr]]:
        """
        Generates a random expression with fresh variables and a replacement
        dictionary that maps fresh variables to x0 ... xi for a synthesis state.
        If the generated variables have different sizes, they are automatically up/downcasted.

        For example, it generates the expression grammar_var1 + grammar_var2
        and the dictionary {grammar_var1: x, grammar_var2: y}. If grammar_var1
        is smaller than grammar_var2, it upcasts grammar_var1 or downcasts
        grammar_var2.

        Returns:
            Tuple[Expr, Dict[Expr, Expr]]: Random expression and dictionary of replacements.
        """
        # generate fresh variables
        v1 = self.gen_fresh_var()
        v2 = self.gen_fresh_var()

        # assign as arguments
        arg1 = v1
        arg2 = v2

        # if arguments are of different size
        if arg1.size != arg2.size:
            # assign the smaller argument to arg1
            if arg1.size > arg2.size:
                arg1, arg2 = arg2, arg1

            # random coin
            coin = getrandbits(32) % 2
            # upcast arg1
            if coin == 0:
                arg1 = arg1.zeroExtend(arg2.size)
            # downcast arg2
            else:
                arg2 = ExprSlice(arg2, 0, arg1.size)

        # sizes have to be equal
        if arg1.size != arg2.size:
            raise RuntimeError(
                f"Argument sizes diverged after coercion: {arg1.size} vs {arg2.size}"
            )

        # flip random coin
        coin = getrandbits(32) % 9

        # generate expression
        if coin == 0:
            expr, num_vars = arg1 + arg2, 2
        elif coin == 1:
            expr, num_vars = arg1 - arg2, 2
        elif coin == 2:
            expr, num_vars = arg1 * arg2, 2
        elif coin == 3:
            expr, num_vars = arg1 & arg2, 2
        elif coin == 4:
            expr, num_vars = arg1 | arg2, 2
        elif coin == 5:
            expr, num_vars = arg1 ^ arg2, 2
        elif coin == 6:
            expr, num_vars = arg1 << arg2, 2
        elif coin == 7:
            expr, num_vars = -v1, 1
        elif coin == 8:
            expr, num_vars = ~v1, 1
        else:
            raise NotImplementedError()

        return expr, self.gen_replacements(num_vars, [v1, v2])

    def gen_terminal_for_state(self) -> Tuple[Expr, Dict[Expr, Expr]]:
        """
        Generates a random terminal expression and a
        replacement dictionary for a synthesis state.

        Returns:
            Tuple[Expr, Dict[Expr, Expr]]: Terminal variable and replacement dictionary.
        """
        # random variable
        variable = self.gen_fresh_var()
        return variable, self.gen_replacements(1, [variable])

    def gen_replacements(
        self, num_vars: int, fresh_variables: List[Expr]
    ) -> Dict[Expr, Expr]:
        """
        Generates a dictionary of replacements. For each fresh variable,
        it chooses a variable x0 ... xi of the same size randomly.

        Args:
            num_vars (int): Number of fresh variables in the expression
            fresh_variables (List[Expr]): List of fresh variables.

        Returns:
            Dict[Expr, Expr]: Dictionary that maps fresh variables to x0 ... xi.
        """
        replacements = {}
        for variable_index in range(num_vars):
            fresh_variable = fresh_variables[variable_index]
            replacements[fresh_variable] = self.get_rand_var_of_size(
                fresh_variable.size
            )

        return replacements
