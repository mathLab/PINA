import argparse

def _init_parser():
    return argparse.ArgumentParser(description="Run PINA")

def setup_generic_run_parser(parser=None):
    if not parser:
        parser = _init_parser()

    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("id_run", help="Run ID", type=int)

    return parser

def setup_extra_features_parser(parser=None):
    if not parser:
        parser = _init_parser()

    parser.add_argument("--extra", help="Extra features", action="store_true")

    return parser
