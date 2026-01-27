import re
from typing import Callable, List, Dict
from miasm.expression.expression import Expr, ExprCond, ExprId, ExprInt, ExprOp, ExprSlice, ExprCompose

def gen_unification_dict(expr: Expr) -> Dict[Expr, Expr]:
    """
    Generates a dictionary of unificiation variables.

    For each unification candidate (terminal expressions such 
    as registers or memory), we generate placeholder variables
    p<index> of the corresponding terminal expression size.

    The resulting dictionary maps termial expressions to their 
    corresponding unification.

    Args:
        expr: Expression to generate unification variables for.

    Returns:
        Dictionary of expressions; terminals are mapped to unification variables.
    """
    return {
        # {x: p0, y: p1, ...,}
        unique_var: ExprId(f"p{index}", unique_var.size)
        for index, unique_var in enumerate(get_unification_candidates(expr))
    }

def get_unique_variables(expr: Expr) -> List[Expr]:
    """
    Get all unique variables in an expression.

    To provide deterministic behavior, the list is sorted.

    Args:
        expr: Expression to parse.

    Returns:
        Sorted list of unique variables.
    """
    l = set()

    def add_to_set(e: Expr) -> Expr:
        """
        Helper function to add variables to a set.

        Args:
            expr: Expression to add.

        Returns:
            Expression.
        """
        if e.is_id():
            l.add(e)
        return e

    expr.visit(add_to_set)

    return sorted(l, key=lambda x: str(x))


def get_unification_candidates(expr: Expr) -> List[Expr]:
    """
    Get all unification candidates in an expression.

    A unification candidate is a leaf in an abstract 
    syntax tree (variable, memory or label). Integers 
    are excluded.

    To provide deterministic behavior, the list is sorted.

    Args:
        expr: Expression to parse.

    Returns:
        Sorted list of unification candidates.
    """
    results = set()

    def add_to_set(e: Expr) -> Expr:
        """
        Helper function to add variables, memory 
        and labels to a set.

        Args:
            expr: Expression to add.

        Returns:
            Expression.
        """
        # memory
        if e.is_mem():
            results.add(e)
        # registers
        if e.is_id():
            results.add(e)
        # location IDs
        if e.is_loc():
            results.add(e)
        return e

    expr.visit(add_to_set)

    return sorted(list(results), key=lambda x: str(x))


def get_subexpressions(expr: Expr) -> List[Expr]:
    """
    Get all subexpressions in descending order.

    This can be understood as a breadth-first search
    on an abstract syntax tree from top to bottom.

    Args:
        expr: Expr

    Returns:
        List of expressions.
    """
    l = []

    def add_to_list(e: Expr) -> Expr:
        """
        Helper function to an expression
        to a list.

        Args:
            expr: Expression to add.

        Returns:
            Expression.
        """
        l.append(e)
        return e

    expr.visit(add_to_list)

    return list(reversed(l))


def compile_expr_to_python(expr: Expr) -> Callable[[List[int]], int]:
    """
    Compile a miasm expression to Python function.

    Instead of walking the expression tree for each evaluation,
    this compiles to a Python lambda that does direct arithmetic.

    Args:
        expr: Miasm expression to compile.

    Returns:
        A function that takes an inputs array and returns the result.
    """
    mask = (1 << expr.size) - 1

    var_map: Dict[int, str] = {}  # expr id -> variable name
    lines: List[str] = []

    def sign_ext(val: str, size: int) -> str:
        sign_bit = 1 << (size - 1)
        return f"(({val}^{sign_bit})-{sign_bit})"

    def compile_node(e: Expr) -> str:
        eid = id(e)

        # check if already assigned to variable
        if eid in var_map:
            return var_map[eid]

        code = compile_node_inner(e)

        # if non-trivial, assign to a variable to avoid evaluating more than once
        if not isinstance(e, (ExprInt, ExprId)):
            var_name = f"_v{len(lines)}"
            lines.append(f"{var_name}={code}")
            var_map[eid] = var_name
            return var_name

        return code

    def compile_node_inner(e: Expr) -> str:
        if isinstance(e, ExprInt):
            return str(int(e))

        elif isinstance(e, ExprId):
            name = e.name
            if name.startswith("p") and name[1:].isdigit():
                idx = int(name[1:])
                var_mask = (1 << e.size) - 1 # Mask input to variable's declared size
                return f"(i[{idx}]&{var_mask})"
            else:
                raise ValueError(f"Unknown variable: {name}")

        elif isinstance(e, ExprOp):
            op = e.op
            args = e.args
            op_mask = (1 << e.size) - 1 # Mask for this operation's result size

            if len(args) == 1:
                a = compile_node(args[0])
                if op == "-":
                    return f"((-{a})&{op_mask})"
                elif op == "parity":
                    # x86 parity: 1 if even number of bits in low 8 bits, 0 if odd
                    return f"((bin(({a})&0xFF).count('1')+1)&1)"
                elif op == "cnttrailzeros":
                    size = e.size
                    return f"(({a}&-{a}).bit_length()-1 if {a} else {size})"
                elif op == "cntleadzeros":
                    size = e.size
                    return f"({size}-({a}).bit_length() if {a} else {size})"
                elif (m := re.fullmatch(r"zeroExt_(\d+)", op)):
                    target_size = int(m.group(1))
                    target_mask = (1 << target_size) - 1
                    return f"(({a})&{target_mask})"
                elif (m := re.fullmatch(r"signExt_(\d+)", op)):
                    target_size = int(m.group(1))
                    source_size = args[0].size
                    target_mask = (1 << target_size) - 1
                    sign_bit = 1 << (source_size - 1)
                    # If sign bit is set, extend with 1s; otherwise keep as-is
                    extension_bits = ((1 << target_size) - 1) ^ ((1 << source_size) - 1)
                    return f"((({a})|{extension_bits})&{target_mask} if ({a})&{sign_bit} else ({a})&{target_mask})"
                else:
                    raise ValueError(f"Unknown unary op: {op}")

            elif op in ["+", "*", "&", "|", "^"] and len(args) >= 2:
                compiled_args = [compile_node(arg) for arg in args]
                py_op = op
                result = py_op.join(compiled_args)
                return f"(({result})&{op_mask})"

            elif len(args) == 2:
                a, b = compile_node(args[0]), compile_node(args[1])
                if op == "-":
                    return f"(({a}-{b})&{op_mask})"
                elif op == ">>":
                    size = e.size
                    return f"((({a})>>({b})&{op_mask}) if ({b})<{size} else 0)"
                elif op == "<<":
                    size = e.size
                    return f"((({a})<<({b})&{op_mask}) if ({b})<{size} else 0)"
                elif op == "a>>":
                    size = e.size
                    sign_bit = 1 << (size - 1)
                    sa = sign_ext(a, size)
                    # Sign extend, then shift; if shift >= size, result is all 1s or 0s based on sign
                    return f"(({sa}>>({b})&{op_mask}) if ({b})<{size} else ({op_mask} if ({a})&{sign_bit} else 0))"
                elif op == "/":
                    return f"(({a}//{b})&{op_mask} if {b} else 0)"
                elif op == "%":
                    return f"(({a}%{b})&{op_mask} if {b} else 0)"
                elif op == "==":
                    return f"(1 if {a}=={b} else 0)"
                elif op == "<u":
                    return f"(1 if {a}<{b} else 0)"
                elif op == "<=u":
                    return f"(1 if {a}<={b} else 0)"
                elif op == "<s":
                    size = e.args[0].size
                    return f"(1 if {sign_ext(a, size)}<{sign_ext(b, size)} else 0)"
                elif op == "<=s":
                    size = e.args[0].size
                    return f"(1 if {sign_ext(a, size)}<={sign_ext(b, size)} else 0)"
                elif op == "<<<":
                    # Rotate left: (x << n) | (x >> (size - n))
                    size = e.size
                    return f"(((({a})<<(({b})%{size}))|(({a})>>(({size}-({b})%{size})%{size})))&{op_mask})"
                elif op == ">>>":
                    # Rotate right: (x >> n) | (x << (size - n))
                    size = e.size
                    return f"(((({a})>>(({b})%{size}))|(({a})<<(({size}-({b})%{size})%{size})))&{op_mask})"
                elif op == "udiv":
                    return f"(({a}//{b})&{op_mask} if {b} else 0)"
                elif op == "umod":
                    return f"(({a}%{b})&{op_mask} if {b} else 0)"
                elif op == "sdiv":
                    size = e.args[0].size
                    sa, sb = sign_ext(a, size), sign_ext(b, size)
                    # Truncate toward zero using integer math to avoid float precision loss.
                    return (
                        f"(((abs({sa})//abs({sb}))"
                        f"*(1 if (({sa})>=0)==(({sb})>=0) else -1))"
                        f"&{op_mask} if {b} else 0)"
                    )
                elif op == "smod":
                    # Result has same sign as dividend: a - trunc(a/b) * b
                    size = e.args[0].size
                    sa, sb = sign_ext(a, size), sign_ext(b, size)
                    return (
                        f"(({sa}-"
                        f"((abs({sa})//abs({sb}))"
                        f"*(1 if (({sa})>=0)==(({sb})>=0) else -1))"
                        f"*{sb})&{op_mask} if {b} else 0)"
                    )
                elif op == "**":
                    return f"(({a}**{b})&{op_mask})"
                else:
                    raise ValueError(f"Unknown binary op: {op}")
            else:
                raise ValueError(f"Unexpected arity {len(args)} for op {op}")

        elif isinstance(e, ExprSlice):
            arg = compile_node(e.arg)
            start, stop = e.start, e.stop
            slice_mask = (1 << (stop - start)) - 1
            return f"(({arg}>>{start})&{slice_mask})"

        elif isinstance(e, ExprCond):
            cond = compile_node(e.cond)
            src1 = compile_node(e.src1)
            src2 = compile_node(e.src2)
            return f"({src1} if {cond} else {src2})"

        elif isinstance(e, ExprCompose):
            result_parts = []
            offset = 0
            for arg in e.args:
                part = compile_node(arg)
                if offset == 0:
                    result_parts.append(f"({part})")
                else:
                    result_parts.append(f"(({part})<<{offset})")
                offset += arg.size
            return "(" + "|".join(result_parts) + ")"

        else:
            raise ValueError(f"Unknown expression type: {type(e).__name__}")

    # build root expression
    result_code = compile_node(expr)
    lines.append(f"return ({result_code})&{mask}")

    # build function src and eval
    func_code = f"def _f(i):{';'.join(lines)}"
    local_ns = {}
    exec(func_code, {}, local_ns)
    return local_ns["_f"]
