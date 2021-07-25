import logging
from argparse import ArgumentParser
from pathlib import Path

from miasm.analysis.binary import Container
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
from miasm.ir.symbexec import SymbolicExecutionEngine

from msynth import Simplifier


logger = logging.getLogger("msynth")


def setup_logging() -> None:
    """Setup logger"""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(name)-18s - %(levelname)-8s: %(message)s'))
    logger.addHandler(console_handler)

def main(file_path: Path, start_addr: int, oracle_path: Path) -> None:
    # symbol table
    loc_db = LocationDB()

    # open the binary for analysis
    container = Container.from_stream(open(file_path, 'rb'), loc_db)

    # cpu abstraction
    machine = Machine(container.arch)

    # init disassemble engine
    mdis = machine.dis_engine(container.bin_stream, loc_db=loc_db)

    # initialize intermediate representation
    lifter = machine.lifter_model_call(mdis.loc_db)

    # disassemble the function at address
    asm_block = mdis.dis_block(start_addr)
    
    # lift to Miasm IR
    ira_cfg = lifter.new_ircfg()
    lifter.add_asmblock_to_ircfg(asm_block, ira_cfg)

    # init symbolic execution engine
    sb = SymbolicExecutionEngine(lifter)

    # symbolically execute basic block
    sb.run_block_at(ira_cfg, start_addr)

    # initialize simplifier
    simplifier = Simplifier(oracle_path)

    for k, v in sb.modified():

        if v.is_int() or v.is_id() or v.is_loc():
            continue
        
        print(f"before: {v}")
        simplified = simplifier.simplify(v)
        print(f"simplified: {simplified}")
        print("\n\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="Example script showing symbolic simplification")
    parser.add_argument("file_path", type=Path, help="Path to binary file")
    parser.add_argument("start_addr", type=str, help="Start address from which to run symbolic execution")
    parser.add_argument("oracle_path", type=Path, help="Path to (serialized) precomputed oracle")
    args = parser.parse_args()
    setup_logging()
    main(args.file_path, int(args.start_addr, 0), args.oracle_path)
