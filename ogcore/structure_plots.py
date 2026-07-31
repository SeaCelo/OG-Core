"""
------------------------------------------------------------------------
Diagrams of a model's structure: which agents exist in a given
parameterization and which flows connect them.

Unlike the other plotting modules, nothing here needs a solved model. Every
diagram is built from a `Specifications` object alone, so a calibration can
be inspected before it is ever run.

Two kinds of information go into the diagrams:

1. The institutional topology -- households supply labor and savings, only
   the Mth industry can supply investment, government spending and debt,
   bequests recirculate within households -- is the model's theory. It is
   declared once, in `get_structure` and the fixed layout of
   `plot_circular_flow`, and changes only when the equations in the theory
   documentation change.

2. The wiring is read from the parameterization: how many industries and
   consumption goods there are, the `io_matrix` coefficients connecting
   them, which tax instruments are switched on, how open the capital and
   debt markets are, and whether remittances, foreign aid, infrastructure
   investment or a UBI exist at all. A channel that is switched off does
   not appear in the diagram.

The same structure feeds several renderers, for different uses:

* `plot_circular_flow`      -- a figure for documents and slides
* `plot_io_heatmap`         -- the industry-to-good coefficients as a matrix
* `plot_calibration_status` -- which values a country calibrated, and which
                               it inherited from OG-Core
* `structure_to_mermaid`    -- text, for documentation kept in version
                               control. Defaults to the institutional graph
                               laid out by ELK: the industries collapse to one
                               node, because every industry repeats the same
                               factor and tax edges, and ELK routes what is
                               left orthogonally. Pass `bundle=False` for one
                               node per industry, and `layout=None` for output
                               that renders on GitHub without a plugin.

`plot_calibration_fit` sits alongside them and is the one figure here that
looks at results rather than inputs: it draws the model-versus-target table a
country calibration reports, from moments the caller supplies.

`make_visuals` produces any or all of them in one call, which is how a
country's set is generated on demand.
------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

# Flow categories. Color and line style both encode the category, so the
# diagrams stay readable in grayscale.
FLOW_STYLES = {
    "real": {"color": "#1D9E75", "linestyle": "-", "label": "Real flows"},
    "fiscal": {"color": "#D85A30", "linestyle": "-", "label": "Fiscal flows"},
    "foreign": {
        "color": "#888780",
        "linestyle": (0, (4, 3)),
        "label": "Cross-border flows",
    },
}

# Fill and edge color for each kind of node.
NODE_STYLES = {
    "household": {"fc": "#EEEDFE", "ec": "#534AB7", "tc": "#26215C"},
    "industry": {"fc": "#EEEDFE", "ec": "#534AB7", "tc": "#26215C"},
    "good": {"fc": "#E1F5EE", "ec": "#0F6E56", "tc": "#04342C"},
    "market": {"fc": "#E1F5EE", "ec": "#0F6E56", "tc": "#04342C"},
    "government": {"fc": "#FAECE7", "ec": "#993C1D", "tc": "#4A1B0C"},
    "foreign": {"fc": "#F1EFE8", "ec": "#5F5E5A", "tc": "#2C2C2A"},
}


# Which parameters govern each box in the structural diagrams. Used to report
# how much of a box rests on country-calibrated values and how much is still
# inherited from OG-Core's own defaults. The market boxes are clearing
# conditions rather than calibrated objects, so they hold no parameters.
PARAM_BLOCKS = {
    "household": [
        "S",
        "J",
        "lambdas",
        "e",
        "beta_annual",
        "sigma",
        "frisch",
        "chi_n",
        "chi_b",
        "g_y_annual",
        "ltilde",
    ],
    "industry": [
        "M",
        "gamma",
        "epsilon",
        "Z",
        "delta_annual",
        "gamma_g",
        "tau_b",
        "delta_tau_annual",
    ],
    "good": ["I", "alpha_c", "io_matrix", "tau_c", "c_min"],
    "government": [
        "alpha_G",
        "alpha_T",
        "alpha_I",
        "initial_debt_ratio",
        "r_gov_scale",
        "r_gov_shift",
        "tau_bq",
        "tau_p",
        "h_wealth",
        "m_wealth",
        "p_wealth",
        "pension_system",
    ],
    "foreign": [
        "zeta_K",
        "zeta_D",
        "alpha_RM_1",
        "alpha_RM_T",
        "alpha_FA",
        "world_int_rate_annual",
        "initial_foreign_debt_ratio",
    ],
}

# Fill colors for the calibration status of a parameter, used where status is
# the subject of the figure.
STATUS_COLORS = {
    "calibrated": "#1D9E75",
    "default": "#EF9F27",
    "missing": "#D3D1C7",
}

# Muted versions of the same, for the chip rows tucked inside the boxes of a
# structural diagram, where calibration is a secondary annotation.
STATUS_TINTS = {
    "calibrated": "#96D3BB",
    "default": "#F7D296",
    "missing": "#E4E2DA",
}

# Text colors for a symbol printed on one of those tints.
STATUS_INK = {
    "calibrated": "#085041",
    "default": "#633806",
    "missing": "#5F5E5A",
}

# The symbol each parameter goes by in the theory documentation, so a chip
# says which parameter it stands for and not merely whether it was
# calibrated. Parameters with no established symbol get a short abbreviation.
PARAM_SYMBOLS = {
    "S": "$S$",
    "J": "$J$",
    "lambdas": r"$\lambda_j$",
    "e": "$e$",
    "beta_annual": r"$\beta$",
    "sigma": r"$\sigma$",
    "frisch": r"$\nu$",
    "chi_n": r"$\chi^n$",
    "chi_b": r"$\chi^b$",
    "g_y_annual": "$g_y$",
    "ltilde": r"$\tilde{l}$",
    "M": "$M$",
    "gamma": r"$\gamma$",
    "epsilon": r"$\varepsilon$",
    "Z": "$Z$",
    "delta_annual": r"$\delta$",
    "gamma_g": r"$\gamma_g$",
    "tau_b": r"$\tau_b$",
    "delta_tau_annual": r"$\delta_\tau$",
    "I": "$I$",
    "alpha_c": r"$\alpha_c$",
    "io_matrix": r"$\pi$",
    "tau_c": r"$\tau_c$",
    "c_min": "$c_{min}$",
    "alpha_G": r"$\alpha_G$",
    "alpha_T": r"$\alpha_T$",
    "alpha_I": r"$\alpha_I$",
    "initial_debt_ratio": "$D/Y$",
    "r_gov_scale": "$r_{sc}$",
    "r_gov_shift": "$r_{sh}$",
    "tau_bq": r"$\tau_{bq}$",
    "tau_p": r"$\tau_p$",
    "h_wealth": "$h_w$",
    "m_wealth": "$m_w$",
    "p_wealth": "$p_w$",
    "pension_system": "pens",
    "zeta_K": r"$\zeta_K$",
    "zeta_D": r"$\zeta_D$",
    "alpha_RM_1": r"$\alpha_{RM}^1$",
    "alpha_RM_T": r"$\alpha_{RM}^T$",
    "alpha_FA": r"$\alpha_{FA}$",
    "world_int_rate_annual": "$r^*$",
    "initial_foreign_debt_ratio": "$D_f/Y$",
}


def _differs(a, b):
    """
    Whether two parameter values differ in substance rather than in shape.

    Comparison broadcasts, so gaining a dimension is not by itself a
    difference: a country that moves to eight industries but gives every one
    of them OG-Core's single elasticity has not calibrated that elasticity,
    and it reads as inherited. Only a value that actually changes counts.
    """
    if isinstance(a, str) or isinstance(b, str):
        return a != b
    try:
        arr_a = np.atleast_1d(np.asarray(a, dtype=float))
        arr_b = np.atleast_1d(np.asarray(b, dtype=float))
    except (TypeError, ValueError):
        return a != b
    try:
        return not np.allclose(
            arr_a, arr_b, rtol=1e-9, atol=1e-12, equal_nan=True
        )
    except ValueError:
        # Shapes that will not broadcast against each other at all.
        return True


def calibration_status(p):
    """
    Report which structural parameters carry country-calibrated values and
    which still hold the value OG-Core ships.

    A parameter counts as calibrated when its value differs from OG-Core's,
    and only then. Writing a parameter into a country's own file does not
    make it calibrated if the number written is the one OG-Core already used;
    neither does spreading that number across a new set of industries. What
    matters is whether the model is running on the country's evidence.

    Comparing values rather than parameter names also catches parameters a
    country sets indirectly. `tau_b`, for instance, is derived from
    `cit_rate`, so the name a country writes down is not the name the model
    ends up carrying.

    Args:
        p (OG-Core Specifications object): model parameters

    Returns:
        status (dict): parameter name mapped to "calibrated", "default" or
            "missing", the last meaning the parameter is absent from this
            version of OG-Core
        blocks (dict): block name mapped to (n_calibrated, n_present)
    """
    from ogcore.parameters import Specifications

    names = [n for block in PARAM_BLOCKS.values() for n in block]
    reference = Specifications()

    status = {}
    for name in names:
        if not hasattr(p, name):
            status[name] = "missing"
        elif _differs(getattr(p, name), getattr(reference, name, None)):
            status[name] = "calibrated"
        else:
            status[name] = "default"

    blocks = {}
    for block, block_names in PARAM_BLOCKS.items():
        present = [
            n for n in dict.fromkeys(block_names) if status[n] != "missing"
        ]
        blocks[block] = (
            sum(status[n] == "calibrated" for n in present),
            len(present),
        )
    return status, blocks


def _active(value, tol=0.0):
    """
    Whether a channel is switched on anywhere in a parameter's time path.

    Args:
        value (array_like or scalar): parameter value
        tol (scalar): magnitude below which the channel counts as off

    Returns:
        (bool): True if any element exceeds tol in absolute value
    """
    if value is None:
        return False
    return bool(
        np.any(np.abs(np.atleast_1d(np.asarray(value, dtype=float))) > tol)
    )


def _base(value):
    """Base-year value of a scalar or time-path parameter."""
    return float(np.atleast_1d(np.asarray(value, dtype=float)).flatten()[0])


def _labels(p, industry_names, good_names):
    """
    Resolve industry and consumption good labels, falling back to generic
    names when a country package has not supplied its own.
    """
    if industry_names is None:
        industry_names = [f"Industry {m + 1}" for m in range(p.M)]
    if good_names is None:
        good_names = [f"Good {i + 1}" for i in range(p.I)]
    if len(industry_names) != p.M:
        raise ValueError(
            f"Got {len(industry_names)} industry names, need M={p.M}"
        )
    if len(good_names) != p.I:
        raise ValueError(f"Got {len(good_names)} good names, need I={p.I}")
    return list(industry_names), list(good_names)


def get_structure(p, industry_names=None, good_names=None, io_threshold=0.01):
    """
    Read a parameterization and return the graph of the model's linkages.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries. Country packages
            keep these in a `PROD_DICT`.
        good_names (list): labels for the I consumption goods, kept in a
            country package's `CONS_DICT`.
        io_threshold (scalar): omit `io_matrix` links below this share, so a
            dense bridge matrix stays legible

    Returns:
        nodes (dict): node id mapped to a dict of label, group and detail
        edges (list): dicts of source, target, label and kind, where kind is
            a key of FLOW_STYLES
    """
    industry_names, good_names = _labels(p, industry_names, good_names)

    nodes, edges = {}, []

    def node(nid, label, group, detail=""):
        nodes[nid] = {"label": label, "group": group, "detail": detail}

    def edge(src, tgt, label, kind):
        edges.append(
            {"source": src, "target": tgt, "label": label, "kind": kind}
        )

    open_K = _active(p.zeta_K)
    open_D = _active(p.zeta_D)
    remittances = _active(getattr(p, "alpha_RM_1", 0.0)) or _active(
        getattr(p, "alpha_RM_T", 0.0)
    )
    aid = _active(getattr(p, "alpha_FA", 0.0))
    infrastructure = _active(getattr(p, "alpha_I", 0.0))

    node("HH", "Households", "household", f"{p.S} ages, {p.J} types")
    node("CAP", "Capital market", "market", "labor and capital")
    node("GOV", "Government", "government", "taxes, transfers, debt")
    if open_K or open_D or remittances or aid:
        node("ROW", "Rest of world", "foreign", "")

    for m, name in enumerate(industry_names):
        node(f"IND{m}", name, "industry", "")
    for i, name in enumerate(good_names):
        node(f"GOOD{i}", name, "good", "")

    # Factor supply. Households own all labor and all domestic savings.
    for m in range(p.M):
        edge("HH", f"IND{m}", "labor", "real")
    edge("HH", "CAP", "savings", "real")
    edge("HH", "HH", "bequests", "real")

    # Capital allocation between domestic and foreign owners.
    for m in range(p.M):
        edge("CAP", f"IND{m}", "domestic capital", "real")
    if open_K:
        for m in range(p.M):
            edge(
                "ROW",
                f"IND{m}",
                f"foreign capital, zeta_K={_base(p.zeta_K):.3g}",
                "foreign",
            )
        edge("CAP", "ROW", "returns on foreign capital", "foreign")
    edge("CAP", "GOV", "debt held at home", "fiscal")
    if open_D:
        edge(
            "ROW",
            "GOV",
            f"debt held abroad, zeta_D={_base(p.zeta_D):.3g}",
            "foreign",
        )
        edge("GOV", "ROW", "debt service abroad", "foreign")

    # The io_matrix bridge: how much of industry m's output composes
    # consumption good i. The one part of the topology that is entirely data.
    io = np.atleast_2d(p.io_matrix)
    for i in range(io.shape[0]):
        for m in range(io.shape[1]):
            if io[i, m] > io_threshold:
                edge(f"IND{m}", f"GOOD{i}", f"{io[i, m]:.2f}", "real")
    alpha_c = np.atleast_1d(p.alpha_c)
    for i in range(p.I):
        edge(f"GOOD{i}", "HH", f"{alpha_c[i]:.2f}", "real")

    # Only the Mth industry's output can be used for investment, government
    # spending, infrastructure and debt.
    last = p.M - 1
    edge(f"IND{last}", "CAP", "investment", "real")
    edge(f"IND{last}", "GOV", "government consumption", "real")
    if infrastructure:
        edge(f"IND{last}", "GOV", "infrastructure investment", "real")
        for m in range(p.M):
            edge("GOV", f"IND{m}", "public capital", "real")

    # One edge per tax instrument that is actually switched on.
    for label, value in (
        ("income tax", True),
        ("payroll tax", getattr(p, "tau_p", 0.0)),
        ("consumption tax", getattr(p, "tau_c", 0.0)),
        ("wealth tax", getattr(p, "p_wealth", 0.0)),
        ("bequest tax", getattr(p, "tau_bq", 0.0)),
    ):
        if value is True or _active(value):
            edge("HH", "GOV", label, "fiscal")
    if _active(getattr(p, "tau_b", 0.0)):
        for m in range(p.M):
            edge(f"IND{m}", "GOV", "corporate income tax", "fiscal")

    # Government outlays.
    edge("GOV", "HH", "transfers", "fiscal")
    if getattr(p, "pension_system", None):
        edge("GOV", "HH", "pensions", "fiscal")
    if _active(getattr(p, "ubi_nom_017", 0.0)) or _active(
        getattr(p, "ubi_nom_1864", 0.0)
    ):
        edge("GOV", "HH", "universal basic income", "fiscal")

    if remittances:
        rm = _base(getattr(p, "alpha_RM_1", 0.0))
        edge("ROW", "HH", f"remittances, {rm:.1%} of GDP", "foreign")
    if aid:
        edge("ROW", "GOV", "foreign aid", "foreign")

    return nodes, edges


def summarize_structure(p, industry_names=None, good_names=None):
    """
    Describe in words which channels a parameterization switches on. Useful
    as a caption beneath any of the diagrams, and as a quick check that a
    calibration contains what its author intended.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries
        good_names (list): labels for the I consumption goods

    Returns:
        (dict): channel name mapped to a short description, or to None when
            the channel is switched off
    """
    industry_names, good_names = _labels(p, industry_names, good_names)
    zK, zD = _base(p.zeta_K), _base(p.zeta_D)
    return {
        "industries": f"M = {p.M}: " + ", ".join(industry_names),
        "consumption goods": f"I = {p.I}: " + ", ".join(good_names),
        "households": f"{p.S} ages by {p.J} lifetime income types",
        "private capital market": (
            f"foreigners supply {zK:.0%} of excess demand"
            if _active(p.zeta_K)
            else "closed, households supply all capital"
        ),
        "government debt market": (
            f"foreigners hold {zD:.0%} of new issues"
            if _active(p.zeta_D)
            else "closed, households hold all debt"
        ),
        "consumption tax": (
            f"{_base(p.tau_c):.1%}"
            if _active(getattr(p, "tau_c", 0.0))
            else None
        ),
        "bequest tax": (
            f"{_base(p.tau_bq):.1%}"
            if _active(getattr(p, "tau_bq", 0.0))
            else None
        ),
        "wealth tax": (
            f"{_base(p.p_wealth):.3g}"
            if _active(getattr(p, "p_wealth", 0.0))
            else None
        ),
        "payroll tax": (
            f"{_base(p.tau_p):.1%}"
            if _active(getattr(p, "tau_p", 0.0))
            else None
        ),
        "corporate income tax": (
            f"{_base(p.tau_b):.1%}"
            if _active(getattr(p, "tau_b", 0.0))
            else None
        ),
        "infrastructure investment": (
            f"{_base(p.alpha_I):.1%} of GDP"
            if _active(getattr(p, "alpha_I", 0.0))
            else None
        ),
        "remittances": (
            f"{_base(p.alpha_RM_1):.1%} of GDP"
            if _active(getattr(p, "alpha_RM_1", 0.0))
            else None
        ),
        "foreign aid": (
            f"{_base(p.alpha_FA):.1%} of GDP"
            if _active(getattr(p, "alpha_FA", 0.0))
            else None
        ),
        "pension system": getattr(p, "pension_system", None),
    }


def _chip_grid(ax, x, y, w, status, chip_w_max=3.4, chip_h=1.45):
    """
    Draw the calibration chips along the bottom of a box, one per parameter,
    each labelled with the parameter's symbol.

    Columns are chosen from the width available, so a wide box gets one row
    and a narrow one wraps. Returns the vertical band consumed, which the
    caller keeps clear of its own text.

    Args:
        status (list): (parameter name, status) pairs, in reading order

    Returns:
        (float): height taken up, in axis units
    """
    pad, col_gap, row_gap = 1.2, 0.3, 0.32
    track = w - 2 * pad
    max_cols = max(1, int((track + col_gap) // (2.6 + col_gap)))
    n_rows = int(np.ceil(len(status) / max_cols))
    cols = int(np.ceil(len(status) / n_rows))
    chip_w = min(chip_w_max, (track - col_gap * (cols - 1)) / cols)

    grid_w = cols * chip_w + col_gap * (cols - 1)
    x0 = x + (w - grid_w) / 2
    for k, (name, state) in enumerate(status):
        row, col = divmod(k, cols)
        cx = x0 + col * (chip_w + col_gap)
        cy = y + 0.8 + (n_rows - 1 - row) * (chip_h + row_gap)
        ax.add_patch(
            FancyBboxPatch(
                (cx, cy),
                chip_w,
                chip_h,
                boxstyle="round,pad=0,rounding_size=0.18",
                linewidth=0,
                facecolor=STATUS_TINTS[state],
                zorder=5,
            )
        )
        ax.text(
            cx + chip_w / 2,
            cy + chip_h / 2,
            PARAM_SYMBOLS.get(name, name),
            ha="center",
            va="center",
            fontsize=6.2,
            color=STATUS_INK[state],
            zorder=6,
        )
    return 0.8 + n_rows * chip_h + (n_rows - 1) * row_gap + 0.6


def _box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    group,
    title_size=11,
    sub_size=9,
    status=None,
    rounding=3,
):
    """
    Draw one labelled node box, returning its bounding box.

    When `status` is a list of (parameter name, status) pairs, a grid of small
    chips sits along the bottom edge, one per governing parameter, each
    carrying the parameter's symbol and tinted by whether it was calibrated.
    The chips use muted tints so they read as a footnote on the box rather
    than a second subject competing with the flows. Border style is left
    alone deliberately, because dashes already mean a cross-border flow.
    """
    style = NODE_STYLES[group]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=0.8,
            facecolor=style["fc"],
            edgecolor=style["ec"],
            zorder=3,
        )
    )
    band = _chip_grid(ax, x, y, w, status) if status else 0.0
    if status:
        n_cal = sum(state == "calibrated" for _, state in status)
        ax.text(
            x + w - 1.0,
            y + h - 1.0,
            f"{n_cal}/{len(status)}",
            ha="right",
            va="top",
            fontsize=7,
            color="#5F5E5A",
            zorder=5,
        )
    # Text is centred in whatever is left above the chip grid.
    base, inner = y + band, h - band
    if subtitle:
        ax.text(
            x + w / 2,
            base + inner * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=style["tc"],
            zorder=4,
        )
        ax.text(
            x + w / 2,
            base + inner * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color=style["ec"],
            zorder=4,
        )
    else:
        ax.text(
            x + w / 2,
            base + inner / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=style["tc"],
            zorder=4,
        )
    return (x, y, w, h)


def _arrow(ax, path, kind, lw=1.4, label="", label_xy=None, label_ha="left"):
    """
    Draw a flow arrow along a list of (x, y) waypoints.

    Args:
        label (str): optional text for the flow
        label_xy (tuple): where to put that text, in axis coordinates. Given
            explicitly rather than derived from the path, because the space
            beside an arrow is not always the space that is free.
        label_ha (str): horizontal alignment of the label
    """
    style = FLOW_STYLES[kind]
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        last = (x1, y1) == path[-1]
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>" if last else "-",
                mutation_scale=11 if last else 0,
                linewidth=lw,
                color=style["color"],
                linestyle=style["linestyle"],
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )
    if label and label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha=label_ha,
            va="center",
            fontsize=8,
            color=style["color"],
            zorder=6,
        )


def plot_circular_flow(
    p,
    industry_names=None,
    good_names=None,
    show_calibration=True,
    include_title=False,
    path=None,
):
    """
    The model's institutional linkages as a circular flow: households and
    industries either side, the goods and factor markets between them,
    government in the middle, and the rest of the world along the bottom.

    The layout is fixed because the institutions are. What varies with the
    parameterization is the detail in each box and which of the cross-border
    and fiscal arrows are drawn at all.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries
        good_names (list): labels for the I consumption goods
        show_calibration (bool): tuck a row of small chips into each box, one
            per governing parameter, coloured by whether it carries a
            country-calibrated value or an OG-Core default
        include_title (bool): whether to include a title on the figure
        path (str): path to save figure to

    Returns:
        fig (Matplotlib figure): the figure, if path is None
    """
    industry_names, good_names = _labels(p, industry_names, good_names)

    if show_calibration:
        status, _ = calibration_status(p)
        blocks = {
            block: [
                (n, status[n])
                for n in dict.fromkeys(names)
                if status[n] != "missing"
            ]
            for block, names in PARAM_BLOCKS.items()
        }
    else:
        blocks = {}

    open_K = _active(p.zeta_K)
    open_D = _active(p.zeta_D)
    remittances = _active(getattr(p, "alpha_RM_1", 0.0)) or _active(
        getattr(p, "alpha_RM_T", 0.0)
    )
    aid = _active(getattr(p, "alpha_FA", 0.0))
    show_row = open_K or open_D or remittances or aid

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    ax.grid(False)

    taxes = [
        label
        for label, value in (
            ("income", True),
            ("payroll", getattr(p, "tau_p", 0.0)),
            ("consumption", getattr(p, "tau_c", 0.0)),
            ("wealth", getattr(p, "p_wealth", 0.0)),
            ("bequest", getattr(p, "tau_bq", 0.0)),
        )
        if value is True or _active(value)
    ]

    _box(
        ax,
        33,
        60,
        34,
        9,
        "Goods market",
        f"I = {p.I} consumption good" + ("s" if p.I > 1 else ""),
        "market",
        status=blocks.get("good"),
    )
    _box(
        ax,
        4,
        34,
        25,
        17,
        "Households",
        f"{p.S} ages, {p.J} types",
        "household",
        status=blocks.get("household"),
    )
    _box(
        ax,
        33,
        34,
        34,
        13,
        "Government",
        ", ".join(taxes) + " taxes",
        "government",
        status=blocks.get("government"),
    )
    _box(
        ax,
        71,
        34,
        25,
        17,
        "Industries",
        f"M = {p.M} sector" + ("s" if p.M > 1 else ""),
        "industry",
        status=blocks.get("industry"),
    )
    _box(ax, 33, 18, 34, 9, "Factor market", "labor and capital", "market")
    if show_row:
        _box(
            ax,
            4,
            4,
            92,
            8,
            "Rest of world",
            "",
            "foreign",
            status=blocks.get("foreign"),
        )

    # Goods flow clockwise across the top, factors back along the bottom.
    _arrow(ax, [(83.5, 51), (83.5, 64.5), (67.5, 64.5)], "real")
    _arrow(ax, [(33, 64.5), (16.5, 64.5), (16.5, 51.5)], "real")
    _arrow(ax, [(16.5, 34), (16.5, 22.5), (32.5, 22.5)], "real")
    _arrow(ax, [(67, 22.5), (83.5, 22.5), (83.5, 33.5)], "real")

    # Government exchanges with each private agent.
    _arrow(ax, [(29, 44), (32.5, 44)], "fiscal")
    _arrow(ax, [(33, 38), (29.5, 38)], "fiscal")
    _arrow(ax, [(71, 44), (67.5, 44)], "fiscal")
    _arrow(ax, [(67, 38), (70.5, 38)], "fiscal")

    # Cross-border arrows run up from the rest of the world into whichever
    # agent actually receives the flow. Foreign debt purchases are routed
    # around the factor market so the arrow lands on the government.
    if show_row:
        if remittances:
            _arrow(
                ax,
                [(8, 12), (8, 33.5)],
                "foreign",
                label=f"remittances\n{_base(p.alpha_RM_1):.1%} of GDP",
                label_xy=(10.5, 15.5),
            )
        if open_D:
            _arrow(
                ax,
                [(31, 12), (31, 30), (40, 30), (40, 33.5)],
                "foreign",
                label=f"{_base(p.zeta_D):.0%} of new debt issues",
                label_xy=(33, 15.5),
            )
        if aid:
            _arrow(
                ax,
                [(60, 12), (60, 30), (52, 30), (52, 33.5)],
                "foreign",
                label=f"foreign aid, {_base(p.alpha_FA):.1%} of GDP",
                label_xy=(62, 15.5),
            )
        if open_K:
            _arrow(
                ax,
                [(90, 12), (90, 33.5)],
                "foreign",
                label=(
                    f"foreign capital\n{_base(p.zeta_K):.0%} of excess demand"
                ),
                label_xy=(88, 16),
                label_ha="right",
            )

    handles = [
        Patch(
            facecolor="none",
            edgecolor=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
        for kind, style in FLOW_STYLES.items()
        if kind != "foreign" or show_row
    ]
    if show_calibration:
        handles += [
            Patch(
                facecolor=STATUS_TINTS["calibrated"],
                label="Country-calibrated parameters",
            ),
            Patch(
                facecolor=STATUS_TINTS["default"],
                label="Left at OG-Core default",
            ),
        ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3 if not show_calibration else 5,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.2,
    )
    if include_title:
        ax.set_title("Model structure", fontsize=13)

    if path is None:
        return fig
    fig_path = os.path.join(path)
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_io_heatmap(
    p,
    industry_names=None,
    good_names=None,
    include_title=False,
    path=None,
):
    """
    The `io_matrix` as a matrix of coefficients: how much of each production
    industry's output composes each consumption good.

    Preferred over any node-and-edge rendering of the same coefficients,
    because a matrix has no layout to crowd. It stays readable for any number
    of industries, shows exact values, and makes structural zeros obvious.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries
        good_names (list): labels for the I consumption goods
        include_title (bool): whether to include a title on the figure
        path (str): path to save figure to

    Returns:
        fig (Matplotlib figure): the figure, if path is None
    """
    industry_names, good_names = _labels(p, industry_names, good_names)
    io = np.atleast_2d(p.io_matrix)

    fig, ax = plt.subplots(figsize=(1.1 * p.M + 3.2, 0.7 * p.I + 2.4))
    ax.grid(False)
    im = ax.imshow(io, cmap="BuGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(p.M))
    ax.set_xticklabels(industry_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(p.I))
    ax.set_yticklabels(good_names, fontsize=9)
    ax.set_xlabel("Production industry", fontsize=10)
    ax.set_ylabel("Consumption good", fontsize=10)

    for i in range(p.I):
        for m in range(p.M):
            share = io[i, m]
            if share < 0.005:
                continue
            ax.text(
                m,
                i,
                f"{share:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if share > 0.55 else "#04342C",
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Share of consumption good")
    if include_title:
        ax.set_title("Input-output bridge matrix", fontsize=13)

    if path is None:
        return fig
    fig_path = os.path.join(path)
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


def _wrap_param(name, width=11):
    """Break a parameter name at underscores so it fits inside a tile."""
    lines, current = [], ""
    for part in name.split("_"):
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def plot_calibration_status(
    p,
    include_title=False,
    path=None,
):
    """
    Which structural parameters a calibration actually set, and which still
    hold the value OG-Core ships, laid out block by block.

    Read this alongside `plot_circular_flow`: the flow diagram shows which
    channels exist, this shows how much of each channel rests on the
    country's own evidence. A block full of defaults is not necessarily
    wrong, but it is a deliberate choice that should be defensible.

    Args:
        p (OG-Core Specifications object): model parameters
        include_title (bool): whether to include a title on the figure
        path (str): path to save figure to

    Returns:
        fig (Matplotlib figure): the figure, if path is None
    """
    status, blocks = calibration_status(p)

    captions = {
        "household": "Households",
        "industry": "Industries",
        "good": "Goods and consumption",
        "government": "Government",
        "foreign": "Rest of world",
    }
    rows = []
    for block, block_names in PARAM_BLOCKS.items():
        names = [
            n for n in dict.fromkeys(block_names) if status[n] != "missing"
        ]
        rows.append((block, names))
    n_cols = max(len(names) for _, names in rows)

    # A left margin of its own so the block captions never run over a tile.
    margin = 4.6
    fig, ax = plt.subplots(
        figsize=(0.86 * (n_cols + margin) + 0.6, 0.95 * len(rows) + 1.5)
    )
    ax.set_xlim(-margin, n_cols + 0.2)
    ax.set_ylim(-1.4, len(rows))
    ax.axis("off")
    ax.grid(False)

    for r, (block, names) in enumerate(rows):
        y = len(rows) - 1 - r
        n_cal, n_present = blocks[block]
        ax.text(
            -0.35,
            y + 0.16,
            captions[block],
            ha="right",
            va="center",
            fontsize=10,
            color=NODE_STYLES[block]["tc"],
        )
        ax.text(
            -0.35,
            y - 0.20,
            f"{n_cal} of {n_present} calibrated",
            ha="right",
            va="center",
            fontsize=8,
            color="#5F5E5A",
        )
        for c, name in enumerate(names):
            ax.add_patch(
                FancyBboxPatch(
                    (c + 0.06, y - 0.34),
                    0.88,
                    0.68,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    linewidth=0,
                    facecolor=STATUS_COLORS[status[name]],
                    alpha=0.85 if status[name] == "calibrated" else 0.95,
                    zorder=2,
                )
            )
            ax.text(
                c + 0.5,
                y,
                _wrap_param(name),
                ha="center",
                va="center",
                fontsize=6.6,
                linespacing=1.25,
                color="white" if status[name] == "calibrated" else "#412402",
                zorder=3,
            )

    ax.legend(
        handles=[
            Patch(
                facecolor=STATUS_COLORS["calibrated"],
                label="Country-calibrated",
            ),
            Patch(facecolor=STATUS_COLORS["default"], label="OG-Core default"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    if include_title:
        ax.set_title("Calibration coverage by block", fontsize=13)

    if path is None:
        return fig
    fig_path = os.path.join(path)
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_calibration_fit(
    moments,
    tolerances=(0.01, 0.05),
    value_format="{:.4g}",
    include_title=False,
    path=None,
):
    """
    How close a calibration lands to the moments it targets.

    This is the figure form of the model-versus-target table a country
    calibration reports. The table's real content is a set of distances, and
    distances are easier to judge as positions than as pairs of numbers, so
    the values stay on the page and the gap between them becomes the
    horizontal axis. A moment on target sits on the centre line; the bands
    say how far off is far.

    Deviation is relative to the target, because the moments are on
    incompatible scales -- a bequest yield of 0.0007 and a wealth-to-GDP
    ratio of 3.35 cannot share a linear axis of levels.

    Args:
        moments (DataFrame or list): one row per targeted moment, with keys
            `name`, `model` and `target`, and optionally `group` to band the
            rows into blocks and `source` for the authority behind the
            target. Rows keep the order given.
        tolerances (tuple): the two relative deviations at which a moment
            stops counting as on target and then as close, used for the
            shaded bands and the dot colors
        value_format (str): format for the model and target columns
        include_title (bool): whether to include a title on the figure
        path (str): path to save figure to

    Returns:
        fig (Matplotlib figure): the figure, if path is None
    """
    import pandas as pd

    df = pd.DataFrame(moments).copy()
    for column in ("name", "model", "target"):
        if column not in df:
            raise ValueError(f"moments needs a '{column}' column")
    if "group" not in df:
        df["group"] = ""
    if "source" not in df:
        df["source"] = ""

    target = df["target"].astype(float)
    model = df["model"].astype(float)
    # Relative where the target is non-zero, absolute where it is not.
    scale = target.abs().where(target.abs() > 0, 1.0)
    df["deviation"] = (model - target) / scale

    near, far = sorted(tolerances)

    def band(dev):
        if abs(dev) <= near:
            return "on"
        return "close" if abs(dev) <= far else "off"

    df["band"] = df["deviation"].apply(band)
    band_color = {
        "on": "#1D9E75",
        "close": "#EF9F27",
        "off": "#D85A30",
    }

    # Lay the rows out top to bottom, leaving a gap where a group changes.
    rows, y, previous = [], 0.0, None
    for _, row in df.iterrows():
        if previous is not None and row["group"] != previous:
            y -= 0.75
        rows.append((y, row))
        previous = row["group"]
        y -= 1.0
    height = -y

    worst = float(df["deviation"].abs().max())
    limit = max(far * 1.6, worst * 1.7, 0.02)

    # Column anchors in the left panel. Model and target are right-aligned so
    # their digits line up; the source runs left from its own anchor.
    x_name, x_model, x_target, x_source = 0.35, 0.50, 0.64, 0.67

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.6, 0.30 * height + 1.9),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.03},
    )
    for ax in (left, right):
        ax.set_ylim(y - 0.4, 2.7)
        ax.grid(False)

    left.set_xlim(0, 1)
    left.axis("off")

    for x, header in ((x_model, "model"), (x_target, "target")):
        left.text(
            x,
            1.05,
            header,
            ha="right",
            va="center",
            fontsize=8.5,
            color="#5F5E5A",
        )
    left.text(
        x_source,
        1.05,
        "source",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#5F5E5A",
    )

    seen_groups = set()
    for row_y, row in rows:
        if row["group"] and row["group"] not in seen_groups:
            seen_groups.add(row["group"])
            left.text(
                0.0,
                row_y + 0.72,
                row["group"],
                ha="left",
                va="center",
                fontsize=9,
                color="#26215C",
            )
        left.text(
            x_name,
            row_y,
            row["name"],
            ha="right",
            va="center",
            fontsize=9,
            color="#131f25",
        )
        left.text(
            x_model,
            row_y,
            value_format.format(float(row["model"])),
            ha="right",
            va="center",
            fontsize=8.5,
            color="#131f25",
        )
        left.text(
            x_target,
            row_y,
            value_format.format(float(row["target"])),
            ha="right",
            va="center",
            fontsize=8.5,
            color="#5F5E5A",
        )
        if row["source"]:
            left.text(
                x_source,
                row_y,
                str(row["source"]),
                ha="left",
                va="center",
                fontsize=7.5,
                color="#8a8a84",
            )

    # A symmetric log axis, because deviations are heavy-tailed: one moment
    # off by a factor of ten would otherwise flatten every other moment onto
    # the centre line and hide the tolerance bands entirely. The region inside
    # the tighter tolerance stays linear.
    right.set_xscale("symlog", linthresh=near, linscale=0.9)
    right.set_xlim(-limit, limit)
    # Tick the tolerance edge and then whole decades. Ticking inside the
    # linear region as well would pile labels on top of each other.
    ticks = [0.0, -far, far]
    if limit < 0.5:
        ticks += [-near, near]
    decade = 1.0
    while decade <= limit:
        ticks += [-decade, decade]
        decade *= 10.0
    right.set_xticks(sorted(set(ticks)))
    right.axvspan(-far, far, color=band_color["close"], alpha=0.10, zorder=0)
    right.axvspan(-near, near, color=band_color["on"], alpha=0.16, zorder=0)
    right.axvline(0, color="#5F5E5A", linewidth=0.8, zorder=1)

    for row_y, row in rows:
        dev = float(row["deviation"])
        color = band_color[row["band"]]
        right.plot(
            [0, dev],
            [row_y, row_y],
            color=color,
            linestyle="-",
            linewidth=1.2,
            zorder=2,
        )
        right.plot(
            [np.clip(dev, -limit * 0.985, limit * 0.985)],
            [row_y],
            marker="o",
            markersize=6,
            color=color,
            zorder=3,
        )
        if abs(dev) > near:
            right.text(
                np.clip(dev, -limit * 0.985, limit * 0.985)
                + limit * (0.035 if dev >= 0 else -0.035),
                row_y,
                f"{dev:+.1%}",
                ha="left" if dev >= 0 else "right",
                va="center",
                fontsize=7.5,
                color=color,
            )

    right.set_yticks([])
    for spine in ("left", "right", "top"):
        right.spines[spine].set_visible(False)
    right.spines["bottom"].set_visible(True)
    right.spines["bottom"].set_color("#D3D1C7")
    right.tick_params(axis="x", labelsize=8, colors="#5F5E5A")
    right.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}" if v else "0")
    right.set_xlabel(
        f"Deviation from target\nshaded bands: {near:.0%} and {far:.0%}",
        fontsize=9,
        color="#5F5E5A",
        linespacing=1.5,
    )

    # The bands carry their own caption on the axis label, rather than a
    # legend or in-plot labels, both of which collide once the linear region
    # is narrow.
    n_on = int((df["band"] == "on").sum())
    right.text(
        0,
        1.9,
        f"{n_on} of {len(df)} moments within {near:.0%} of target",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#131f25",
    )

    if include_title:
        fig.suptitle("Calibration fit to targeted moments", fontsize=13)

    if path is None:
        return fig
    fig_path = os.path.join(path)
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


_MERMAID_GROUPS = [
    ("household", "Households"),
    ("industry", "Production industries"),
    ("good", "Consumption goods"),
    ("market", "Markets"),
    ("government", "Government"),
    ("foreign", "Rest of world"),
]


def make_visuals(
    p,
    which="all",
    industry_names=None,
    good_names=None,
    moments=None,
    output_dir=None,
    prefix="",
    options=None,
):
    """
    Generate one, several or all of this module's visuals in a single call.

    The intended use is on demand for a country: hand it a parameterization
    and it writes the set. Every visual except `calibration_fit` is built from
    the parameterization alone, so the usual call needs nothing else.

    Args:
        p (OG-Core Specifications object): model parameters
        which (str or iterable): "all", or a name or names from VISUALS
        industry_names (list): labels for the M industries. Country packages
            keep these in a `PROD_DICT`.
        good_names (list): labels for the I consumption goods, from a
            `CONS_DICT`.
        moments (DataFrame or list): the targeted moments, needed only for
            `calibration_fit`. Requesting that visual without them is an
            error; asking for "all" without them skips it, since the targets
            come from published sources rather than from the parameters.
        output_dir (str): directory to write into. When None every visual is
            returned in memory: figures as Matplotlib figures, `mermaid` as
            its source text.
        prefix (str): prepended to each filename, for tagging a country
        options (dict): visual name mapped to keyword arguments for it, for
            example {"mermaid": {"bundle": False}}

    Returns:
        (dict): visual name mapped to a file path when `output_dir` is given,
            and to the figure or source text when it is not
    """
    if isinstance(which, str):
        names = list(VISUALS) if which == "all" else [which]
    else:
        names = list(which)

    unknown = [n for n in names if n not in VISUALS]
    if unknown:
        raise ValueError(
            f"Unknown visual(s) {unknown}. Choose from {list(VISUALS)}."
        )

    if "calibration_fit" in names and moments is None:
        if which == "all":
            names.remove("calibration_fit")
        else:
            raise ValueError(
                "calibration_fit needs `moments`, the model-versus-target "
                "table; it cannot be derived from the parameters."
            )

    options = options or {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name in names:
        kwargs = dict(options.get(name, {}))
        if name == "mermaid":
            text = structure_to_mermaid(
                p, industry_names, good_names, **kwargs
            )
            if output_dir is None:
                results[name] = text
            else:
                path = os.path.join(output_dir, f"{prefix}{name}.mmd")
                with open(path, "w") as f:
                    f.write(text)
                results[name] = path
            continue

        func = VISUALS[name]
        if name == "calibration_fit":
            args = (moments,)
        elif name == "calibration_status":
            args = (p,)
        else:
            args = (p,)
            kwargs.setdefault("industry_names", industry_names)
            kwargs.setdefault("good_names", good_names)

        if output_dir is None:
            results[name] = func(*args, **kwargs)
        else:
            path = os.path.join(output_dir, f"{prefix}{name}.png")
            func(*args, path=path, **kwargs)
            results[name] = path

    return results


def bundle_group_edges(nodes, edges):
    """
    Collapse an edge repeated across a whole group into one edge on the group.

    Every industry hires labor, rents domestic and foreign capital, uses
    public capital and pays corporate income tax, so `get_structure` emits one
    edge per industry for each of those. That is faithful, and for a model
    with eight industries it is forty near-identical edges that no automatic
    layout can place without piling the labels on top of each other. Since
    they say the same thing about every member of the group, one edge drawn
    on the group says it once.

    A fan-out is only collapsed when it reaches *every* member of the group.
    Reaching some of them is a real distinction and is left alone, otherwise
    the diagram would claim a channel that does not exist. The industry-to-good
    coefficients are never collapsed: each carries its own share as a label, so
    they never group.

    Args:
        nodes (dict): as returned by `get_structure`
        edges (list): as returned by `get_structure`

    Returns:
        (list): edges with each whole-group fan-out replaced by a single edge
            whose endpoint is the group name, carrying `bundled` with the
            number of edges it stands for. Original order is preserved.
    """
    members = {}
    for nid, meta in nodes.items():
        members.setdefault(meta["group"], set()).add(nid)
    groupable = {g for g, ms in members.items() if len(ms) > 1}

    def group_of(nid):
        return nodes[nid]["group"]

    result = [dict(e) for e in edges]
    for direction in ("out", "in"):
        buckets = {}
        for idx, e in enumerate(result):
            if e is None or e["source"] == e["target"] or "bundled" in e:
                continue
            anchor, spread = (
                (e["source"], e["target"])
                if direction == "out"
                else (e["target"], e["source"])
            )
            group = group_of(spread)
            if group not in groupable or group_of(anchor) == group:
                continue
            key = (anchor, e["label"], e["kind"], group)
            buckets.setdefault(key, []).append(idx)

        for (anchor, label, kind, group), idxs in buckets.items():
            reached = {
                result[i]["target"]
                if direction == "out"
                else result[i]["source"]
                for i in idxs
            }
            if reached != members[group]:
                continue
            first = min(idxs)
            result[first] = {
                "source": anchor if direction == "out" else group,
                "target": group if direction == "out" else anchor,
                "label": label,
                "kind": kind,
                "bundled": len(idxs),
            }
            for i in idxs:
                if i != first:
                    result[i] = None

    return [e for e in result if e is not None]


# What an edge becomes once both of its groups collapse to single nodes and
# several edges land on the same pair. The coefficients themselves are left to
# `plot_io_heatmap`, which shows them far better than any label can.
SUMMARY_LABELS = {
    ("industry", "good"): "output",
    ("good", "household"): "consumption",
}


def collapse_group_nodes(nodes, edges, groups, captions):
    """
    Replace each named group with a single node standing for all its members.

    The companion to `bundle_group_edges`, and the reason it is needed:
    bundling strips the factor and tax edges off the individual industries, so
    most of them end up carrying no edge at all. Listing them separately then
    adds nodes an automatic layout has no reason to place anywhere in
    particular, and they drift to the far corners of the diagram.

    Edges that ran between two members of the same collapsed group would
    become self-loops and are dropped. Edges that collapse onto the same pair
    are merged, taking a summary label from SUMMARY_LABELS.

    Args:
        nodes (dict): as returned by `get_structure`
        edges (list): as returned by `get_structure`, ideally already bundled
        groups (iterable): group names to collapse
        captions (dict): group name mapped to its display caption

    Returns:
        nodes (dict), edges (list): with the named groups collapsed
    """
    groups = set(groups)
    members = {g: [] for g in groups}
    for nid, meta in nodes.items():
        if meta["group"] in groups:
            members[meta["group"]].append(nid)

    collapsed = {}
    for nid, meta in nodes.items():
        if meta["group"] not in groups:
            collapsed[nid] = meta
    dimension = {"industry": "M", "good": "I"}
    for group in groups:
        if not members[group]:
            continue
        count = len(members[group])
        symbol = dimension.get(group)
        collapsed[group] = {
            "label": captions.get(group, group),
            "group": group,
            "detail": f"{symbol} = {count}" if symbol else str(count),
        }

    def remap(nid):
        # An endpoint may already be a group name, if the edge came out of
        # `bundle_group_edges`.
        if nid in members:
            return nid
        meta = nodes.get(nid)
        if meta is None:
            return nid
        group = meta["group"]
        return group if group in groups and members[group] else nid

    merged, order = {}, []
    for idx, e in enumerate(edges):
        source, target = remap(e["source"]), remap(e["target"])
        if source == target and e["source"] != e["target"]:
            continue
        # Only edges the collapse actually moved are candidates for merging.
        # Two that already ran between the same pair carry distinct labels
        # worth keeping, and Mermaid draws them side by side happily.
        moved = source != e["source"] or target != e["target"]
        key = (source, target, e["kind"]) if moved else (idx,)
        if key in merged:
            merged[key].append(e)
        else:
            merged[key] = [e]
            order.append((key, source, target, e["kind"]))

    out = []
    for key, source, target, kind in order:
        group = merged[key]
        labels = list(dict.fromkeys(e["label"] for e in group))
        summary = SUMMARY_LABELS.get(
            (
                nodes.get(group[0]["source"], {}).get("group"),
                nodes.get(group[0]["target"], {}).get("group"),
            )
        )
        joined = ", ".join(labels)
        if len(labels) == 1:
            label = labels[0]
        elif summary:
            # A named summary beats a list of coefficients, which say nothing
            # without knowing which industry or good each belongs to.
            label = summary
        elif len(joined) <= 58:
            label = joined
        else:
            label = f"{len(group)} flows"
        out.append(
            {"source": source, "target": target, "label": label, "kind": kind}
        )
    return collapsed, out


def structure_to_mermaid(
    p,
    industry_names=None,
    good_names=None,
    io_threshold=0.01,
    bundle=True,
    layout="elk",
    link_width=2,
):
    """
    The structure as Mermaid flowchart text. GitHub renders this natively in
    markdown, and Jupyter Book renders it with the sphinxcontrib-mermaid
    extension, so it suits documentation that should stay in version control.

    Mermaid lays the graph out on its own, and it does that badly with many
    industries, because every industry repeats the same five factor and tax
    edges. `bundle` is the answer: it collapses those fan-outs onto the group
    and the group onto one node, leaving the institutional structure. The
    industry-by-industry coefficients are then read off `plot_io_heatmap`,
    which shows them better than a label ever could.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries
        good_names (list): labels for the I consumption goods
        io_threshold (scalar): omit coefficients below this share
        bundle (bool): produce the institutional graph, with whole-group
            fan-outs and the industry and consumption-good groups collapsed.
            Set False for the full graph, one node per industry and one edge
            per coefficient, which is faithful but crowded above a few
            industries.
        layout (str): the Mermaid layout engine. Defaults to "elk", which
            routes edges orthogonally and brings each one into its node
            square-on, so a path is far easier to follow than under the
            default engine's long curves. ELK needs the
            `@mermaid-js/layout-elk` plugin registered in the renderer:
            mermaid.live, the VS Code extension and a local build have it,
            GitHub's own markdown rendering does not. Pass None for output
            that renders anywhere.
        link_width (scalar): stroke width for the edges, in px

    Returns:
        (str): Mermaid flowchart source
    """
    nodes, edges = get_structure(p, industry_names, good_names, io_threshold)
    if bundle:
        edges = bundle_group_edges(nodes, edges)
        nodes, edges = collapse_group_nodes(
            nodes, edges, ("industry", "good"), dict(_MERMAID_GROUPS)
        )
    lines = []
    if layout:
        lines.append('%%{init: {"layout": "' + layout + '"}}%%')
    lines.append("flowchart LR")
    for group, caption in _MERMAID_GROUPS:
        members = [n for n, v in nodes.items() if v["group"] == group]
        if not members:
            continue
        # A group of one needs no wrapper: its caption would just repeat the
        # node's own label.
        grouped = len(members) > 1
        if grouped:
            lines.append(f'    subgraph {group}["{caption}"]')
        for nid in members:
            detail = nodes[nid]["detail"]
            label = nodes[nid]["label"] + (f" ({detail})" if detail else "")
            lines.append(
                f'{"        " if grouped else "    "}{nid}["{label}"]'
            )
        if grouped:
            lines.append("    end")
    for e in edges:
        lines.append(f'    {e["source"]} -- "{e["label"]}" --> {e["target"]}')
    for idx, e in enumerate(edges):
        style = FLOW_STYLES[e["kind"]]
        dash = ",stroke-dasharray:4 3" if e["kind"] == "foreign" else ""
        lines.append(
            f"    linkStyle {idx} stroke:{style['color']},"
            f"stroke-width:{link_width}px{dash}"
        )
    for group, _ in _MERMAID_GROUPS:
        members = [n for n, v in nodes.items() if v["group"] == group]
        if not members:
            continue
        s = NODE_STYLES[group]
        lines.append(
            f"    classDef {group} fill:{s['fc']},"
            f"stroke:{s['ec']},color:{s['tc']}"
        )
        lines.append(f"    class {','.join(members)} {group}")
    return "\n".join(lines)


# The visuals `make_visuals` can produce, in the order it produces them.
# Defined last because it names the functions above it.
VISUALS = {
    "circular_flow": plot_circular_flow,
    "io_heatmap": plot_io_heatmap,
    "calibration_status": plot_calibration_status,
    "calibration_fit": plot_calibration_fit,
    "mermaid": structure_to_mermaid,
}
