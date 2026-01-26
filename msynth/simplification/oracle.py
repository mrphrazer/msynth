import hashlib
import multiprocessing
import pickle
import re
import warnings
import sqlite3
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from miasm.expression.expression import Expr, ExprInt
from miasm.expression.simplifications import expr_simp

from msynth.simplification.ast import AbstractSyntaxTreeTranslator
from msynth.utils.expr_utils import get_unique_variables, compile_expr_to_python, parse_expr
from msynth.utils.sampling import gen_inputs


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
            warnings.warn(f"compile_expr fallback: {e} - consider adding support in compile_expr()")
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

        Args:
            file_path: Path to pre-computed SimplificationOracle file.

        Returns:
            Deserialized SimplificationOracle.
        """
        try:
            return SimplificationOracle._load_from_sqlite(file_path)
        except sqlite3.DatabaseError:
            # Not sqlite, try pickle
            with open(file_path, 'rb') as f:
                oracle: SimplificationOracle = pickle.load(f)
            assert isinstance(oracle, SimplificationOracle), f"Expected SimplificationOracle, found {type(oracle)}"
            return oracle

    @staticmethod
    def _load_from_sqlite(file_path: Path) -> 'SimplificationOracle':
        """Load oracle from sqlite format with lazy loading."""
        conn = sqlite3.connect(file_path)

        # Load metadata
        cur = conn.execute("SELECT key, value FROM metadata")
        meta = dict(cur.fetchall())

        oracle = SimplificationOracle.__new__(SimplificationOracle)
        oracle.num_variables = meta['num_variables']
        oracle.num_samples = meta['num_samples']
        oracle.inputs = pickle.loads(meta['inputs'])
        oracle._conn = conn
        oracle.oracle_map = SqliteOracleMap(conn)
        oracle._runtime_cache = {}
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
            if Path(file_path).exists():
                # Remove existing to rebuild from scratch
                Path(file_path).unlink()

            conn = sqlite3.connect(file_path)
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value BLOB)")
            conn.execute("CREATE TABLE equiv_classes (equiv_class BLOB PRIMARY KEY, members BLOB)")

            conn.execute("INSERT INTO metadata VALUES (?, ?)",
                         ('num_variables', self.num_variables))
            conn.execute("INSERT INTO metadata VALUES (?, ?)",
                         ('num_samples', self.num_samples))
            conn.execute("INSERT INTO metadata VALUES (?, ?)",
                         ('inputs', pickle.dumps(self.inputs)))

            for k, v in self.oracle_map.items():
                conn.execute("INSERT INTO equiv_classes VALUES (?, ?)",
                             (bytes.fromhex(k), SqliteOracleMap._exprs_to_bytes(v)))
            conn.commit()
            conn.close()
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)

    def close(self) -> None:
        """
        Close the underlying sqlite connection if open.
        """
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
            self._conn = None


class SqliteOracleMap:
    """Dict-like lazy-loading proxy for sqlite oracle storage.

    Key hashes are stored as 20 byte binary BLOBs.
    Members are stored as zlib-compressed newline-separated repr strings.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @staticmethod
    def _to_bin(hex_key: str) -> bytes:
        """Convert hex string key to binary for storage."""
        return bytes.fromhex(hex_key)

    @staticmethod
    def _to_hex(bin_key: bytes) -> str:
        """Convert binary key back to hex string."""
        return bin_key.hex()

    @staticmethod
    def _exprs_to_bytes(exprs: List[Expr]) -> bytes:
        """Serialize list of expressions as compressed repr strings."""
        text = '\n'.join(repr(e) for e in exprs)
        return zlib.compress(text.encode(), level=9)

    @staticmethod
    def _bytes_to_exprs(data: bytes) -> List[Expr]:
        """Deserialize compressed repr strings back to expressions."""
        text = zlib.decompress(data).decode()
        return [parse_expr(line) for line in text.split('\n') if line]

    def __contains__(self, key: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM equiv_classes WHERE equiv_class = ?",
            (self._to_bin(key),))
        return cur.fetchone() is not None

    def __getitem__(self, key: str) -> List[Expr]:
        cur = self._conn.execute(
            "SELECT members FROM equiv_classes WHERE equiv_class = ?",
            (self._to_bin(key),))
        row = cur.fetchone()
        if row is None:
            raise KeyError(key)
        return self._bytes_to_exprs(row[0])

    def keys(self) -> Iterator[str]:
        cur = self._conn.execute("SELECT equiv_class FROM equiv_classes")
        return (self._to_hex(row[0]) for row in cur)

    def items(self) -> Iterator[Tuple[str, List[Expr]]]:
        cur = self._conn.execute("SELECT equiv_class, members FROM equiv_classes")
        for row in cur:
            yield self._to_hex(row[0]), self._bytes_to_exprs(row[1])

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM equiv_classes")
        return cur.fetchone()[0]

