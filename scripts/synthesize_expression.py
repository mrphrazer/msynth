import logging
import time
from argparse import ArgumentParser
from msynth import Synthesizer
from miasm.expression.expression import Expr, ExprId, ExprInt

logger = logging.getLogger("msynth")


def setup_logging() -> None:
    """Setup logger"""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(name)-18s - %(levelname)-8s: %(message)s'))
    logger.addHandler(console_handler)

def mba(size: int) -> Expr:
    """Generate exemplary MBA expression (for testing/debug purposes)"""
    v0 = ExprId("v0", size)
    v1 = ExprId("v1", size)
    v2 = ExprId("v2", size)
    return (((v1 ^ v2) + ((v1 & v2) << ExprInt(1, size))) | v0) + (((v1 ^ v2) + ((v1 & v2) << ExprInt(1, size))) & v0)

def main(expr: Expr, num_io_samples: int) -> None:
    start_time = time.time()
    # init synthesizer engine
    logger.debug("Initializing synthesizer")
    simplifier = Synthesizer()
    logger.info("Synthesizing expression")
    # synthesize expression
    synthesized, score = simplifier.synthesize_from_expression(expr, num_samples=num_io_samples)
    print(f"initial: {expr}")
    print(f"synthesized: {synthesized} (score: {score})")
    logger.info(f"Done in {round(time.time() - start_time, 2)} seconds")


if __name__ == "__main__":
    parser = ArgumentParser(description="Simplify expressions (especially MBAs) using stochastic program synthesis")
    parser.add_argument("--samples", dest="num_io_samples", type=int, default=10, 
        help="Number of individual input-output samples for the synthesizer. \
        The smaller the number, the less accurate the results, but also the faster the computation. \
            ~ 10: fast, but less accurate \
            ~ 100: slow, way more accurate: \
            ~ 30 to 50 are good trade offs")
    args = parser.parse_args()
    setup_logging()
    main(mba(32), args.num_io_samples)
