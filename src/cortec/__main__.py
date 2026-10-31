"""The `cortec` command (also `python -m cortec`). `cortec run ...` runs the whole pipeline on your
table (docs/setup.md step 1); `cortec --help` prints what you can run, in what order; `cortec <name>`
prints one function or class with its arguments and defaults read from the code."""
import sys

import cortec
from . import help as _help

RUN_ORDER = [                      # the value column fits the page width after the key column
    ("1  Schema", "declare the columns, bounds, bins and the target (docs/setup.md, section 4)"),
    ("2  release_statistics", "Stage A: spend the budget once on a release (docs/setup.md, section 5)"),
    ("3  Generator", "Stage B: .generate or .generate_selected from it (docs/setup.md, section 6)"),
    ("3b decode_release", "Stage B with no model: Private-PGM plus the selection step (docs/setup.md, section 6)"),
    ("4  bound_with_controls", "Stage C: a DP utility bound with its controls (docs/setup.md, section 7)"),
    ("5  evaluate", "fidelity and utility beside real references (docs/setup.md, section 8)"),
    ("   show, record", "print and export any stage in one layout (docs/setup.md, section 11)"),
    ("   install_guard, guard", "print a refusal in that layout, not a traceback (docs/setup.md, section 10)"),
]

GROUPS = [                         # the other public names, by what they are
    ("results", ["Release", "ReleaseModel", "BoundReport", "BoundResult", "RunRecord", "ResultTable"]),
    ("selection", ["select_to_release", "inclusion_weights", "pool_coverage",
                   "release_subbin_values"]),
    ("models", ["check_model", "validated_models"]),
    ("hierarchy", ["derive", "DerivedConfig"]),
    ("privacy", ["PrivacyLedger", "Band", "transmission_bound", "register_refusal"]),
]


def run(argv):
    """the whole pipeline on your table. cortec run --help for the settings"""
    from .run import main as run_main
    return run_main(argv)


def decode(argv):
    """decode a release with no language model (Private-PGM + the selection step). cortec decode --help"""
    from .decode import main as decode_main
    return decode_main(argv)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _help.main(cortec, argv, tool="cortec", module="cortec", command="cortec",
                      run_order=RUN_ORDER, groups=GROUPS, readme="docs/setup.md",
                      example="example.py", subcommands={"run": run, "decode": decode}, terms=_help.TERMS_CORTEC)


if __name__ == "__main__":
    sys.exit(main())
