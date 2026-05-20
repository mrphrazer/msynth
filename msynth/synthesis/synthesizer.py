import logging
import multiprocessing
import time
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

from miasm.expression.expression import Expr
from msynth.synthesis.grammar import Grammar
from msynth.synthesis.mutations import Mutator
from msynth.synthesis.oracle import SynthesisOracle
from msynth.synthesis.scoring import CompiledCache, score_expression
from msynth.synthesis.smir import SmirEngine, SmirRule, default_smir_rules
from msynth.synthesis.state import SynthesisState
from msynth.utils.expr_utils import get_unique_variables
from msynth.utils.parallelizer import Parallelizer
from msynth.utils.unification import gen_unification_dict, reverse_unification

logger = logging.getLogger("msynth.synthesizer")


@dataclass(frozen=True)
class SearchEvaluation:
    best_expr: Expr
    best_score: float
    best_rule: str
    aggregate_score: float


class RawSearchEngine:
    """
    Scores the raw candidate expression without applying Smir rules.

    This is used for `Synthesizer(use_smir=False)`. It keeps one local-search
    driver in Synthesizer while making the opt-out mean exactly "do not infer
    extra candidates from the current expression".
    """

    def __init__(self):
        self.compiled_cache: CompiledCache = {}

    def evaluate(
        self, expr: Expr, oracle: SynthesisOracle, variables: list[Expr]
    ) -> SearchEvaluation:
        score = score_expression(expr, oracle, variables, self.compiled_cache)
        return SearchEvaluation(
            best_expr=expr,
            best_score=score,
            best_rule="raw",
            aggregate_score=score,
        )


class Synthesizer:
    """
    Expression synthesis based on input-output (I/O) samples.

    Based on a number of I/O samples (provided via a synthesis oracle),
    it tries to stochastically learn an expression with the same I/O behavior.
    For this, it considers program synthesis as a stochastic optimization problem:
    It randomly mutates a synthesis state (expression), measures the differences in
    the output behavior between the synthesis state and the oracle output
    using a distance function. The goal is to minimize the distance until the
    I/O behavior is the same.

    The synthesizer makes use of a grammar to generate random expressions. To mutate
    expressions, it uses a mutator that applies small changes to the synthesis state.

    The approach is inspired by:

    "Syntia: Synthesizing the Semantics of Obfuscated Code" by
    Tim Blazytko, Moritz Contag, Cornelius Aschermann and Thorsten Holz (USENIX Security 2017).
    Link: https://synthesis.to/papers/usenix17-syntia.pdf

    "Search-Based Local Blackbox Deobfuscation: Understand, Improve and Mitigate" by
    Grégoire Menguy, Sébastien Bardin, Richard Bonichon and Cauim de Souza de Lima (CCS 2021).
    Link: https://binsec.github.io/assets/publications/papers/2021-ccs.pdf

    The synthesizer provides the functionality to synthesize an expression based on a
    given (complex) expression that represents a mathematical function f(x0, ..., xi)
    with i inputs. Based on this expression, synthesis oracle, grammar and mutator
    will be automatically initialized. The synthesis can also be processed in parallel.
    """

    def __init__(
        self,
        use_smir: bool = True,
        smir_rules: Sequence[SmirRule] | None = None,
        polynomial_degree: int = 2,
    ):
        """
        Initialize the stochastic synthesizer.

        Args:
            use_smir: Enable Search Modulo Inference Rules by default. When
                disabled, synthesis uses the same local-search driver with raw
                candidate scoring only.
            smir_rules: Optional ordered Smir rule list. The order matters when
                multiple inferred candidates match the oracle with score 0.0.
            polynomial_degree: Maximum polynomial degree for the default Smir
                polynomial rule.
        """
        self.use_smir = use_smir
        self.smir_rules = (
            list(smir_rules)
            if smir_rules is not None
            else default_smir_rules(polynomial_degree)
        )

    def synthesize_from_expression(
        self, expr: Expr, num_samples: int, timeout: Union[int, float] = 60
    ) -> Tuple[Expr, float]:
        """
        Synthesizes an expression from a given expression that represents a function f(x0, ..., xi).

        The function constructs a synthesis oracle, a grammar and a mutator. Afterward, it
        performs the stochastic search. To deal with memory, memory accesses are replaced with
        variable accesses and re-applied afterward to the synthesized expression.

        Example:

        The given expression @64[rax] + rbx - rbx is unified to p0 + p1 - p1. A synthesized
        expression p0 with he same I/O behavior is p0. After re-applying the initial variables,
        we return @64[rax].

        Args:
            expr (Expr): Expression representing a function f(x0, ..., xi) in Miasm IR.
            num_samples (int): Number of I/O samples for the synthesis oracle.
            timeout (int | float): Timeout in seconds for the local search.

        Returns:
            Tuple[Expr, float]: Synthesized expression and its corresponding score.
        """
        # unify expression (to remove memory etc.)
        unification_dict = gen_unification_dict(expr)
        expr = expr.replace_expr(unification_dict)

        # get list of unique variables
        variables = get_unique_variables(expr)

        # generate synthesis oracle
        oracle = SynthesisOracle.gen_from_expression(expr, variables, num_samples)

        # init grammar
        grammar = Grammar(expr.size, variables)

        # build mutator
        mutator = Mutator.gen_from_expression(expr, grammar)

        # perform stochastic search
        if timeout == 60:
            state, score = self.iterated_local_search(mutator, oracle)
        else:
            state, score = self.iterated_local_search(mutator, oracle, timeout=timeout)

        # reverse unification and re-apply original variables
        expr = reverse_unification(state.get_expr_simplified(), unification_dict)

        # upcast expression if necessary
        if grammar.size > expr.size:
            expr = expr.zeroExtend(grammar.size)

        return expr, score

    def synthesize_from_expression_parallel(
        self, expr: Expr, num_samples: int
    ) -> Tuple[Expr, float]:
        """
        Performs the synthesis for a given expression that represents a function f(x0, ..., xi) in parallel.

        The function call to `synthesize_from_expression` is parallelized. All synthesis tasks
        are clustered into the same task group and passed to a Parallelizer. If the first worker from
        the task group succeeds, other instances will be terminated. In case no worker succeeds,
        the initial provided expression will be returned.

        Args:
            expr (Expr): Expression representing a function f(x0, ..., xi) in Miasm IR.
            num_samples (int): Number of I/O samples for the synthesis oracle.

        Returns:
            Tuple[Expr, float]:  Synthesized expression and its corresponding score.
        """

        # parallelization wrapper
        def synthesize_from_expression(results: Any, index: Any) -> None:
            result = self.synthesize_from_expression(expr, num_samples)
            # if synthesis succeeded
            if result[1] == 0.0:
                results[index] = result

        # prepare parallelization
        tasks = []
        task_group = f"{expr}"
        for _ in range(multiprocessing.cpu_count()):
            tasks.append((synthesize_from_expression, task_group))

        # execute in parallel
        parallelizer = Parallelizer(tasks)
        parallelizer.execute()

        # get task group result
        if task_group in parallelizer.task_group_results:
            result = parallelizer.task_group_results[task_group]
            # check if result found
            if result is not None:
                return result[0], result[1]

        return expr, float("inf")

    def simplify(
        self, expr: Expr, num_samples: int = 10, timeout: Union[int, float] = 60
    ) -> Expr:
        """
        Simplifies an expression via stochastic program synthesis.

        If the synthesis was not successful, the initial expression
        is returned.

        Args:
            expr (Expr): Expression to simplify in Miasm IR.
            num_samples (int, optional): Number of I/O samples for the synthesis oracle. Defaults to 10.
            timeout (int | float): Timeout in seconds for the local search.

        Returns:
            Expr: Simplified Expression in Miasm IR.
        """
        # synthesize
        if timeout == 60:
            synthesized, score = self.synthesize_from_expression(expr, num_samples)
        else:
            synthesized, score = self.synthesize_from_expression(
                expr, num_samples, timeout=timeout
            )
        # check if perfect score
        if score == 0.0:
            return synthesized

        return expr

    def iterated_local_search(
        self, mutator: Mutator, oracle: SynthesisOracle, timeout: Union[int, float] = 60
    ) -> Tuple[SynthesisState, float]:
        """
        Performs an iterative local search (ILS) to synthesize an expression for a given synthesis oracle.

        The algorithm tries to find a synthesis state that minimizes the distance function. Starting with an AST
        representing a single leaf, it iteratively switches between perturbation and side search. The perturbation
        mutates the best state (found so far) by replacing a subexpression with a leaf node. Afterward, it tries
        to find better synthesis states in the side search by applying more aggressive mutations. The mutated state is
        discarded, unless it is better than the current state. If the side search does not find a better state
        (with a lower score) within 100 iterations, the algorithm continues with perturbation.

        The algorithm terminates and returns the best state and score if

        (1) the synthesis state's score is 0 (synthesis state and oracle have the same I/O behavior) or
        (2) the provided timeout is reached.

        The implementation is adapted from:

        "Search-Based Local Blackbox Deobfuscation: Understand, Improve and Mitigate" by
        Grégoire Menguy, Sébastien Bardin, Richard Bonichon and Cauim de Souza de Lima (CCS 2021).
        Link: https://binsec.github.io/assets/publications/papers/2021-ccs.pdf

        Args:
            mutator (Mutator): Mutator to mutate synthesis states.
            oracle (SynthesisOracle): Input-output oracle.
            timeout (int, optional): Timeout for synthesis. Defaults to 60.

        Returns:
            Tuple[SynthesisState, float]: Best synthesis state and its score.
        """
        search_engine = (
            SmirEngine(self.smir_rules) if self.use_smir else RawSearchEngine()
        )

        # init states
        current_state = SynthesisState(*mutator.grammar.gen_terminal_for_state())
        current_eval = search_engine.evaluate(
            current_state.get_expr(), oracle, mutator.grammar.variables
        )

        best_expr = current_eval.best_expr
        best_score = current_eval.best_score
        best_rule = current_eval.best_rule
        if best_score == 0.0:
            return SynthesisState(best_expr), best_score

        # init iterations and time
        iteration = 0
        change_counter = 0
        start_time = time.time()

        while time.time() - start_time < timeout:
            iteration += 1

            proposed_state = mutator.mutate(current_state.clone())
            proposed_eval = search_engine.evaluate(
                proposed_state.get_expr(), oracle, mutator.grammar.variables
            )

            # Algorithm 3 from the Smir paper accepts a mutation if the product
            # of inferred-candidate scores improves. After a stagnation window,
            # it accepts a mutation anyway to escape a local minimum.
            if (
                proposed_eval.aggregate_score < current_eval.aggregate_score
                or change_counter >= 100
            ):
                current_state = proposed_state.clone()
                current_eval = proposed_eval
                change_counter = 0

                if proposed_eval.best_score < best_score:
                    best_expr = proposed_eval.best_expr
                    best_score = proposed_eval.best_score
                    best_rule = proposed_eval.best_rule
                    logger.info(
                        f"best Smir expression: {best_expr} "
                        f"(rule: {best_rule}) "
                        f"(score: {best_score}) "
                        f"(iteration: {iteration})"
                    )

                if best_score == 0.0:
                    return SynthesisState(best_expr), best_score
            else:
                change_counter += 1

        return SynthesisState(best_expr), best_score
