import time
from argparse import ArgumentParser
from pathlib import Path
from msynth import Simplifier
from miasm.expression.expression import Expr, ExprId, ExprInt


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
    return ((~v0 | v2) - ~((~((((ExprInt(0x1, size) + v2) - ExprInt(0x1, size)) | ~v0) - ~v0) & v2) + (v2 + (v2 & ~v2)))) + (((v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1))) + (v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))) + (v0 ^ (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))) + ((ExprInt(0x1, size) + ~(((v1 + (~v1 & v2)) | ((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2))) - (v1 & (v1 + (~v1 & v2))))) + ((- (- v0)) + (((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2)) & ~v2))) + ((~v0 | v2) - ~((~((((ExprInt(0x1, size) + v2) - ExprInt(0x1, size)) | ~v0) - ~v0) & v2) + (v2 + (v2 & ~v2)))) + (((v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1))) + (v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))) + (v0 ^ (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))) + ((ExprInt(0x1, size) + ~(((v1 + (~v1 & v2)) | ((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2))) - (v1 & (v1 + (~v1 & v2))))) + ((- (- v0)) + (((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2)) & ~v2))) + ((~v0 | v2) - ~((~((((ExprInt(0x1, size) + v2) - ExprInt(0x1, size)) | ~v0) - ~v0) & v2) + (v2 + (v2 & ~v2)))) + (((v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1))) + (v0 & (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))) + (v0 ^ (((v0 & v1) + (v0 & v1)) + (v0 ^ v1)))) + ((ExprInt(0x1, size) + ~(((v1 + (~v1 & v2)) | ((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2))) - (v1 & (v1 + (~v1 & v2))))) + ((- (- v0)) + (((v1 + (~v1 & v2)) + (~(v1 + (~v1 & v2)) & v2)) & ~v2)))

def main(oracle_path: Path, expr: Expr) -> None:
    start_time = time.time()
    # init simplification engine
    logger.debug("Initializing simplification engine")
    simplifier = Simplifier(oracle_path)
    logger.info("Simplifying expression")
    # simplify expression
    simplified = simplifier.simplify(expr)
    print(f"initial: {expr}")
    print(f"simplified: {simplified}")
    logger.info(f"Done in {round(time.time() - start_time, 2)} seconds")


if __name__ == "__main__":
    parser = ArgumentParser(description="Simplify expressions (especially MBAs) using a precomputed Oracle")
    parser.add_argument("oracle_path", type=Path, help="Path to (serialized) precomputed oracle")
    args = parser.parse_args()
    setup_logging()
    main(args.oracle_path, mba(32))
