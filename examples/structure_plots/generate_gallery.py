"""
Generate the structural and calibration visuals for a country model.

Run it from the country repo's own environment, because it reads that
package's parameter files and its industry and consumption-good labels:

    cd ~/Projects/OG-PHL
    PYTHONPATH=~/Projects/OG-Core .venv/bin/python \
        ~/Projects/OG-Core/examples/structure_plots/generate_gallery.py \
        --package ogphl \
        --params ogphl_default_parameters.json \
        --overlay ogphl_multisector_default_parameters.json \
        --moments ~/Projects/OG-Core/examples/structure_plots/phl_moments.csv \
        --out . --prefix phl_

The figures in this directory were produced by exactly that command, against
the OG-PHL multi-industry calibration.

The Mermaid graph is written as a PNG, which needs the Mermaid CLI
(`npm install -g @mermaid-js/mermaid-cli`). Pass `--mermaid-format mmd` to
write the source instead and skip that requirement.
"""

import argparse
import importlib
import importlib.resources
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ogcore import structure_plots  # noqa: E402
from ogcore.parameters import Specifications  # noqa: E402


def load_json(package, name):
    """Read a JSON parameter file packaged inside a country module."""
    with importlib.resources.files(package).joinpath(name).open() as f:
        return json.load(f)


def build_parameters(package, params, overlay):
    """
    Build a Specifications object from a country's packaged parameters.

    A multi-industry calibration is usually shipped as an overlay holding only
    what it changes, so it has to go on top of the single-industry base rather
    than be loaded alone.
    """
    p = Specifications()
    if params:
        p.update_specifications(load_json(package, params))
    if overlay:
        p.update_specifications(load_json(package, overlay))
    return p


def read_labels(package):
    """
    Read a country package's industry and consumption-good labels.

    Country packages keep these in `constants.PROD_DICT` and
    `constants.CONS_DICT`. Missing either is fine; the visuals then fall back
    to generic names.
    """
    try:
        constants = importlib.import_module(f"{package}.constants")
    except ModuleNotFoundError:
        return None, None
    industries = getattr(constants, "PROD_DICT", None)
    goods = getattr(constants, "CONS_DICT", None)
    return (
        list(industries) if industries else None,
        list(goods) if goods else None,
    )


def read_moments(path):
    """Read the model-versus-target table, if one was given."""
    if not path:
        return None
    import pandas as pd

    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        required=True,
        help="country package to read, for example ogphl",
    )
    parser.add_argument(
        "--params", help="packaged single-industry parameter JSON"
    )
    parser.add_argument(
        "--overlay", help="packaged multi-industry overlay JSON, applied after"
    )
    parser.add_argument(
        "--moments",
        help="CSV of targeted moments with name, model, target and optionally "
        "group and source. Without it the calibration fit is skipped.",
    )
    parser.add_argument("--out", default=".", help="directory to write into")
    parser.add_argument("--prefix", default="", help="filename prefix")
    parser.add_argument(
        "--format", default="png", help="figure format: png, svg, pdf, ..."
    )
    parser.add_argument(
        "--mermaid-format",
        default="png",
        help='"png", "svg", "pdf" via the Mermaid CLI, or "mmd" for source',
    )
    args = parser.parse_args()

    p = build_parameters(args.package, args.params, args.overlay)
    industries, goods = read_labels(args.package)
    # A package's label lists describe its multi-industry setup; when the
    # parameterization being drawn has a different M or I (e.g. the
    # single-industry default of a package that also ships a multi-industry
    # overlay), fall back to generic labels rather than erroring.
    if industries is not None and len(industries) != p.M:
        print(
            f"note: {len(industries)} industry labels for M = {p.M}; "
            "using generic labels"
        )
        industries = None
    if goods is not None and len(goods) != p.I:
        print(
            f"note: {len(goods)} good labels for I = {p.I}; "
            "using generic labels"
        )
        goods = None
    plt.style.use("ogcore.OGcorePlots")

    written = structure_plots.make_visuals(
        p,
        industry_names=industries,
        good_names=goods,
        moments=read_moments(args.moments),
        output_dir=args.out,
        prefix=args.prefix,
        fmt=args.format,
        mermaid_fmt=args.mermaid_format,
    )

    print(f"M = {p.M}, I = {p.I}")
    for name, path in written.items():
        print(f"  {name:20} {os.path.basename(path)}")

    for channel, value in structure_plots.summarize_structure(
        p, industries, goods
    ).items():
        print(f"  {channel:26} {value if value is not None else '(off)'}")


if __name__ == "__main__":
    main()
