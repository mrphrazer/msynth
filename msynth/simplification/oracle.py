import hashlib
import multiprocessing
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from miasm.expression.expression import Expr, ExprInt
from miasm.expression.simplifications import expr_simp

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.utils.expr_utils import get_unique_variables, compile_expr_to_python, parse_expr
from msynth.utils.sampling import gen_inputs
from msynth.utils.sqlite import NotSqliteOracleError, dump_sqlite_oracle_data, load_sqlite_oracle_data


def calc_hash(s: str) -> str:
    """
    Calculates the hash of a equivalence class

    Args:
        s: ASCII string to hash

    Returns:
        SHA1 of input as string
    """
    return hashlib.sha1(s.encode('ascii')).hexdigest()


class SimplificationOracle(object):
    """
    Simplification Oracle used for I/O sampling and equivalence class lookup.

    Intuitively, it models the semantic I/O behavior of all
    mathematical functions f(x0, ..., xi) for `num_variables`
    parameters and `num_samples` input-output samples.

    Each function  f(x0, ..., xi) is approximated by a equivalence
    class based on its outputs [out0, out1, ..., out_num_samples],
    which maps to a list of expressions that form an equivalence class
    (a set of expressions that share the same output behavior for a
    set of inputs).

    To build such a mapping for a pre-computed set of formulas 
    (a so-called library), it takes a path to a library file as input 
    and clusters the individual expressions into equivalence classes.
    Afterward, they can be stored in a file.

    In summary, this class provides the following functionality:

    1. Generation of equivalence classes for a given set of expressions (library)
    2. Serialization and Deserialization
    3. Evaluation of expressions to determine their equivalence class

    Attributes:
        num_variables (int): Number of variables for f(x0, ..., xi)
        num_samples (int): Number of independent I/O pairs to group expressions
        inputs (List[List[int]]): Oracle inputs for I/O sampling.
        oracle_map (Dict[Expr, List[Expr]]): Maps output behavior to a list of expressions.
    """

    def __init__(self, num_variables: int, num_samples: int, library_path: Path):
        """
        Initializes an SimplificationOracle instance.

        Args:
            num_variables: Number of variables.
            num_samples: Number of independent I/O samples.
            library_path: File path to a set of expressions.
        """
        self.num_variables = num_variables
        self.num_samples = num_samples
        self.inputs = gen_inputs(num_variables, num_samples)
        self.oracle_map = self.gen_oracle_map(library_path)

    def get_outputs(self, expr: Expr) -> List[int]:
        """
        Determines the output behavior for a given expression.

        Args:
            expr: Expression to evaluate.

        Returns:
            List of ints, representing the expression' output behavior.
        """
        try:
            func = compile_expr_to_python(expr)
            return [func(input_array) for input_array in self.inputs]
        except ValueError as e:
            # Fallback to slower tree-walking evaluation for unsupported expression types
            return [self.evaluate_expression(expr, input_array) for input_array in self.inputs]

    @staticmethod
    def parse_library(library_path: Path) -> Iterator[str]:
        """
        Opens a pre-computed library file and iterates over the individual expressions.
        The library must contain one Miasm IR expression per line.

        Args:
            library_path: Path to pre-computed library file.

        Returns:
            Iterator over strings containing a Miasm IR expression.
        """
        yield from library_path.read_text().splitlines()

    def determine_equiv_class(self, expr: Expr, outputs: List[int]) -> str:
        """ 
        Determines an expressions' equivalence class.

        The equivalence is a cryptographic string over

        1. the expression's size
        2. the expression's output behavior.

        For example: `hash(32, [0, 1, 2, 3])` for a 32-bit expression
        and a vector of outputs.

        Args:
            expr: Expression to determine equivalence class for.
            outputs: Expression's output behavior.

        Returns:
            String representing the expression's equilavence class.
        """
        # used to prefix expression sizes
        identifier = str(expr.size)
        return calc_hash(identifier + str(outputs))

    def get_equiv_class_members(self, equiv_class: str) -> Iterator[Expr]:
        """
        Iterates over all members for a given equivalence class. 

        Args:
            equiv_class: Equivalence class as string.

        Returns:
            Iterator over equivalence class members.
        """
        if hasattr(self, '_runtime_cache') and equiv_class in self._runtime_cache:
            for member in self._runtime_cache[equiv_class]:
                yield member
        else:
            for member in self.oracle_map[equiv_class]:
                yield member

    def contains_equiv_class(self, equiv_class: str) -> bool:
        """
        Checks if the simplification oracle contains a specific equivalence class.

        Args:
            equiv_class: Equivalence class as string.

        Returns:
            True if equiv class in oracle, False otherwise
        """
        if hasattr(self, '_runtime_cache') and equiv_class in self._runtime_cache:
            return True
        return equiv_class in self.oracle_map

    def set_equiv_class(self, equiv_class: str, members: List[Expr]) -> None:
        """
        Sets the members for an equivalence class.

        For lazy-loaded oracles, this writes to the runtime cache.
        For in-memory oracles, this writes directly to oracle_map.

        Args:
            equiv_class: Equivalence class as string.
            members: List of expressions for this equivalence class.
        """
        if hasattr(self, '_runtime_cache'):
            self._runtime_cache[equiv_class] = members
        else:
            self.oracle_map[equiv_class] = members

    def _expr_str_to_equiv_class(self, expr_str: str) -> Tuple[str, Expr]:
        """
        Determines the equivalence class of a given Miasm IR expression 
        (passed as string).

        Used as part of the parallel computation in `gen_oracle_map`.

        Args:
            expr_str: String containing a Miasm IR expression from the
                      pre-computed library.
        
        Returns:
            Tuple of equivalence class and expression.
        """
        # init AST translator
        translator = AbstractSyntaxTreeTranslator()
        # read expression
        expr = parse_expr(expr_str)
        # simplify and transform into abtsract syntax tree
        expr = translator.from_expr(expr_simp(expr))
        # calculate output behavior
        outputs = self.get_outputs(expr)
        # determine equivalence class
        equiv_class = self.determine_equiv_class(expr, outputs)
        return (equiv_class, expr)

    def gen_oracle_map(self, library_path: Path) -> Dict[str, List[Expr]]:
        """
        Builds the oracle map by clustering expressions into equivalence classes.

        The oracle map is a dictionary that maps equivalence classes to a list
        of semantically equivalent (based on their I/O behavior) expressions.
        The lists are sorted in ascending order by the expressions' depth

        To compute the oracle map, we do the following:

        1. Walk over all expressions in the library and determine their equivalence class
        2. Unify them by storing them in a set
        3. Sort the unique expressions based on their depth.

        Args:
            library_path: Path to pre-computed library of expressions.

        Returns:
            Dictionary that maps equivalence classes to list of expressions.
        """
        # init dictionaries
        oracle_map_tmp: Dict[str, Set[Expr]] = {}
        oracle_map: Dict[str, List[Expr]] = {}

        # process each expression contained in the library in parallel
        expression_strings = SimplificationOracle.parse_library(library_path)
        with multiprocessing.Pool() as pool:
            mapping = pool.map(self._expr_str_to_equiv_class, expression_strings)
        # sort into equivalence classes and unify expressions in library
        for (equiv_class, expr) in mapping:
            # do not add integers to the oracle
            if expr.is_int():
                continue
            # add to set of expressions for equivalence class
            oracle_map_tmp.setdefault(equiv_class, set()).add(expr)

        # prepare final data structure
        for k in oracle_map_tmp.keys():
            # sort from smallest to largest expression (number of nodes)
            oracle_map[k] = list(
                sorted(oracle_map_tmp[k], key=lambda x: len(x.graph().nodes())))

        return oracle_map

    @staticmethod
    def evaluate_expression(expr: Expr, inputs_array: List[int]) -> int:
        """
        Evaluates an expression for an array of random values.

        Each input variable p0, p1, ..., pn is associated with an
        entry in the array of inputs [i0, i1, ..., in]. In the given 
        expression, we replace p0 with i1, p1 with i1 etc. and evaluate
        the expression. As a result, the expression results in a 
        final constant in form of ExprInt.

        Args:
            expr: Expression to evaluate
            inputs_array: List of random values.

        Returns: 
            Int that is the return value of the evaluated expression.
        """
        # dictionary of replacements
        replacements = {}
        # walk over unique variables in the expression
        for v in get_unique_variables(expr):
            # skip if register pattern does not match
            if not re.search("^p[0-9]*", v.name):
                continue
            # calculate index for p
            index = int(v.name.strip("p"))
            # insert into replacements dictionary
            replacements[v] = ExprInt(inputs_array[index], v.size)

        return int(expr_simp(expr.replace_expr(replacements)))

    @staticmethod
    def load_from_file(file_path: Path) -> 'SimplificationOracle':
        """
        Load a pre-computed SimplificationOracle from a given file.

        Supports both pickle format and sqlite format.
        For sqlite format, equivalence classes are queried and loaded lazily
        rather than all at once.

        For sqlite oracles, release database connection when finished by
        calling close() or using context manager:

            with SimplificationOracle.load_from_file(path) as oracle:
                ...

        Args:
            file_path: Path to pre-computed SimplificationOracle file.

        Returns:
            Deserialized SimplificationOracle.
        """
        try:
            meta, conn, oracle_map = load_sqlite_oracle_data(file_path)
            oracle = SimplificationOracle.__new__(SimplificationOracle)
            oracle.num_variables = meta['num_variables']
            oracle.num_samples = meta['num_samples']
            oracle.inputs = meta['inputs']
            oracle._conn = conn
            oracle.oracle_map = oracle_map
            oracle._runtime_cache = {}
            return oracle
        except NotSqliteOracleError:
            # Not sqlite, try pickle
            with open(file_path, 'rb') as f:
                oracle = pickle.load(f)
            if not isinstance(oracle, SimplificationOracle):
                raise TypeError(
                    f"Expected SimplificationOracle, found {type(oracle)}"
                )
            return oracle

    def dump_to_file(self, file_path: Path, use_sqlite: bool = False) -> None:
        """
        Stores a SimplificationOracle instance to a file.

        Args:
            file_path: File path to store object instance.
            use_sqlite: If True, use sqlite format for lazy loading support.
                        If False (default), use pickle format.

        Returns:
            None
        """
        if use_sqlite:
            dump_sqlite_oracle_data(file_path, self.num_variables, self.num_samples, self.inputs, self.oracle_map)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)

    def close(self) -> None:
        """
        Close any resources held by this oracle.

        For sqlite-backed oracles, this closes the database connection.
        For pickle-loaded oracles, this is a no-op.
        """
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> 'SimplificationOracle':
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing any resources."""
        self.close()
