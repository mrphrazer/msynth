import logging
import time
from argparse import ArgumentParser
from pathlib import Path
from msynth import SimplificationOracle

logger = logging.getLogger("msynth")


def setup_logging() -> None:
    """Setup logger"""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(name)-18s - %(levelname)-8s: %(message)s'))
    logger.addHandler(console_handler)

def main(num_variables: int, num_io_samples: int, library_path: Path, oracle_path: Path) -> None:
    start_time = time.time()
    logger.info(f"Computing oracle for {num_variables} variables and {num_io_samples} samples. Using library at '{library_path.as_posix()}'")
    oracle = SimplificationOracle(num_variables, num_io_samples, library_path)
    logger.info(f"Writing oracle to {oracle_path.as_posix()}")
    oracle.dump_to_file(oracle_path)
    logger.info(f"Done in {round(time.time() - start_time, 2)} seconds")


if __name__ == "__main__":
    parser = ArgumentParser(description="Precompute SimplificationOracle which then can be used to simplify MBA expressions")
    parser.add_argument("library_path", type=Path, help="Path to a pre-computed set of expressions that shall be clustered in a oracle")
    parser.add_argument("oracle_path", type=Path, help="Output path to store the calculated oracle")
    parser.add_argument("--variables", dest="num_variables", type=int, default=30, 
        help="Number of variables in a function f(x0, ..., xn). \
        Since msynth sometimes uses intermediate variables, a number of ~30 is mostly a good trade off.")
    parser.add_argument("--samples", dest="num_io_samples", type=int, default=50, 
        help="Number of individual output samples to determine the expression's equivalence class. \
        The smaller the number, the less accurate the results, but also the faster the computation. \
            ~ 10: fast, but less accurate \
            ~ 100: slow, way more accurate: \
            ~ 30 to 50 are good trade offs")
    args = parser.parse_args()
    setup_logging()
    main(args.num_variables, args.num_io_samples, args.library_path, args.oracle_path)
