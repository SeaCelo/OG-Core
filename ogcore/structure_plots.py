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

* `plot_circular_flow`   -- a figure for documents and slides
* `plot_io_bridge`       -- the industry-to-good coefficients, as ribbons
* `plot_io_heatmap`      -- the same coefficients as a matrix, for large M
* `structure_to_mermaid` -- text, renders natively in GitHub markdown
* `structure_to_dot`     -- text for Graphviz, which lays out dense graphs
                            far better than Mermaid
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


def _differs(a, b):
    """Whether two parameter values differ, tolerating shape changes."""
    if isinstance(a, str) or isinstance(b, str):
        return a != b
    try:
        arr_a = np.atleast_1d(np.asarray(a, dtype=float))
        arr_b = np.atleast_1d(np.asarray(b, dtype=float))
    except (TypeError, ValueError):
        return a != b
    if arr_a.shape != arr_b.shape:
        return True
    return not np.allclose(arr_a, arr_b, rtol=1e-9, atol=1e-12, equal_nan=True)


def calibration_status(p, calibrated_params=None):
    """
    Report which structural parameters carry country-calibrated values and
    which still hold the value OG-Core ships.

    A parameter counts as calibrated when its value differs from the one
    OG-Core ships. That alone catches parameters a country sets indirectly:
    `tau_b`, for instance, is derived from `cit_rate`, so the name a country
    writes down is not the name the model carries.

    Passing `calibrated_params` adds a second signal, the names the
    calibration actually set. It settles the case value comparison cannot
    see: a country that deliberately adopted OG-Core's number for a
    parameter did make a choice, and reads as calibrated rather than
    inherited. The two signals are combined, never traded off.

    Args:
        p (OG-Core Specifications object): model parameters
        calibrated_params (set, list or dict): names of the parameters the
            calibration actually set. A dict is read for its keys, so the
            JSON overlays a country package applies can be passed straight
            in. When None, only value comparison is used.

    Returns:
        status (dict): parameter name mapped to "calibrated", "default" or
            "missing", the last meaning the parameter is absent from this
            version of OG-Core
        blocks (dict): block name mapped to (n_calibrated, n_present)
    """
    from ogcore.parameters import Specifications

    names = [n for block in PARAM_BLOCKS.values() for n in block]
    reference = Specifications()

    explicit = set()
    if calibrated_params is not None:
        if isinstance(calibrated_params, dict):
            explicit = set(calibrated_params)
        else:
            for item in calibrated_params:
                explicit.update(item if isinstance(item, dict) else [item])

    status = {}
    for name in names:
        if not hasattr(p, name):
            status[name] = "missing"
        elif name in explicit or _differs(
            getattr(p, name), getattr(reference, name, None)
        ):
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

    When `status` is a list of per-parameter statuses, a row of small chips
    along the bottom edge shows one chip per governing parameter, in the same
    order and reading as the same colors as `plot_calibration_status`, with
    the count in the corner. The chips use muted tints and a capped width:
    they are a footnote on the box, not a second subject competing with the
    flows. Border style is left alone deliberately, because dashes already
    mean a cross-border flow on the arrows.
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
    if status:
        n = len(status)
        n_cal = sum(s == "calibrated" for s in status)
        gap = 0.3
        chip_w = min(1.9, (w - 2.4 - gap * (n - 1)) / n)
        chip_h = min(0.7, h * 0.07)
        row_w = n * chip_w + gap * (n - 1)
        x0 = x + (w - row_w) / 2
        for k, state in enumerate(status):
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + k * (chip_w + gap), y + 0.85),
                    chip_w,
                    chip_h,
                    boxstyle="square,pad=0",
                    linewidth=0,
                    facecolor=STATUS_TINTS[state],
                    zorder=5,
                )
            )
        ax.text(
            x + w - 1.0,
            y + h - 1.0,
            f"{n_cal}/{n}",
            ha="right",
            va="top",
            fontsize=7,
            color="#5F5E5A",
            zorder=5,
        )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.60,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=style["tc"],
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
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
            y + h / 2,
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
    calibrated_params=None,
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
        calibrated_params (set, list or dict): passed to
            `calibration_status`, to name the calibrated parameters exactly
            rather than inferring them
        include_title (bool): whether to include a title on the figure
        path (str): path to save figure to

    Returns:
        fig (Matplotlib figure): the figure, if path is None
    """
    industry_names, good_names = _labels(p, industry_names, good_names)

    if show_calibration:
        status, _ = calibration_status(p, calibrated_params)
        blocks = {
            block: [
                status[n]
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
        f"I = {p.I} consumption goods",
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
        f"M = {p.M} sectors",
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


def plot_io_bridge(
    p,
    industry_names=None,
    good_names=None,
    io_threshold=0.05,
    include_title=False,
    path=None,
):
    """
    The `io_matrix` coefficients as weighted ribbons from each production
    industry to each consumption good. Ribbon width is the coefficient, so
    the industry that dominates a consumption good is visible at a glance.

    Best below roughly ten industries. Above that the ribbons overlap and
    `plot_io_heatmap` reads more clearly.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries
        good_names (list): labels for the I consumption goods
        io_threshold (scalar): omit coefficients below this share
        include_title (bool): whether to include a title on the figure
        path (str): path to save figure to

    Returns:
        fig (Matplotlib figure): the figure, if path is None
    """
    industry_names, good_names = _labels(p, industry_names, good_names)
    io = np.atleast_2d(p.io_matrix)
    alpha_c = np.atleast_1d(p.alpha_c)

    # One unit of height per row, so the geometry does not depend on M or I.
    rows = max(p.M, p.I)
    box_h, pitch = 0.62, 1.0
    top = rows * pitch

    fig, ax = plt.subplots(figsize=(9.6, 0.66 * rows + 1.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.9, top + 0.5)
    ax.axis("off")
    ax.grid(False)

    def centers(n):
        offset = (rows - n) * pitch / 2
        return {k: top - offset - (k + 0.5) * pitch for k in range(n)}

    ind_y, good_y = centers(p.M), centers(p.I)

    order = sorted(
        (
            (io[i, m], i, m)
            for i in range(io.shape[0])
            for m in range(io.shape[1])
            if io[i, m] > io_threshold
        )
    )
    for share, i, m in order:
        # linestyle is pinned because OGcorePlots cycles it, and a dashed
        # ribbon here would read as a different kind of flow.
        ax.plot(
            [31, 67],
            [ind_y[m], good_y[i]],
            color=FLOW_STYLES["real"]["color"],
            linestyle="-",
            alpha=0.30 + 0.45 * share,
            linewidth=0.5 + 7.0 * share,
            solid_capstyle="round",
            zorder=1,
        )

    for m, name in enumerate(industry_names):
        _box(
            ax,
            4,
            ind_y[m] - box_h / 2,
            27,
            box_h,
            name,
            "",
            "industry",
            title_size=9,
            rounding=0.12,
        )
    for i, name in enumerate(good_names):
        lead = io[i].argmax()
        _box(
            ax,
            67,
            good_y[i] - box_h / 2,
            25,
            box_h,
            name,
            "",
            "good",
            title_size=9,
            rounding=0.12,
        )
        ax.text(
            93.5,
            good_y[i] + 0.11,
            f"{alpha_c[i]:.0%} of consumption",
            ha="left",
            va="center",
            fontsize=7.5,
            color=NODE_STYLES["good"]["tc"],
        )
        ax.text(
            93.5,
            good_y[i] - 0.15,
            f"{io[i, lead]:.0%} {industry_names[lead].lower()}",
            ha="left",
            va="center",
            fontsize=7.5,
            color=NODE_STYLES["good"]["ec"],
        )

    ax.text(
        4,
        -0.55,
        "Ribbon width is the share of the consumption good supplied by that "
        f"industry; shares below {io_threshold:.0%} are omitted",
        fontsize=8,
        color="#5F5E5A",
    )
    if include_title:
        ax.set_title("Production industries to consumption goods", fontsize=13)

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
    The `io_matrix` as a matrix of coefficients. Unlike `plot_io_bridge`
    this has no layout to crowd, so it stays readable for any number of
    industries, shows exact values, and makes structural zeros obvious.

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
    calibrated_params=None,
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
        calibrated_params (set, list or dict): passed to
            `calibration_status`, to name the calibrated parameters exactly
            rather than inferring them
        include_title (bool): whether to include a title on the figure
        path (str): path to save figure to

    Returns:
        fig (Matplotlib figure): the figure, if path is None
    """
    status, blocks = calibration_status(p, calibrated_params)

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


_MERMAID_GROUPS = [
    ("household", "Households"),
    ("industry", "Production industries"),
    ("good", "Consumption goods"),
    ("market", "Markets"),
    ("government", "Government"),
    ("foreign", "Rest of world"),
]


def structure_to_mermaid(
    p, industry_names=None, good_names=None, io_threshold=0.01
):
    """
    The structure as Mermaid flowchart text. GitHub renders this natively in
    markdown, and Jupyter Book renders it with the sphinxcontrib-mermaid
    extension, so it suits documentation that should stay in version
    control. Mermaid lays out dense graphs poorly; use `structure_to_dot`
    when the graph is large.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries
        good_names (list): labels for the I consumption goods
        io_threshold (scalar): omit coefficients below this share

    Returns:
        (str): Mermaid flowchart source
    """
    nodes, edges = get_structure(p, industry_names, good_names, io_threshold)
    lines = ["flowchart LR"]
    for group, caption in _MERMAID_GROUPS:
        members = [n for n, v in nodes.items() if v["group"] == group]
        if not members:
            continue
        lines.append(f'    subgraph {group}["{caption}"]')
        for nid in members:
            detail = nodes[nid]["detail"]
            label = nodes[nid]["label"] + (f" ({detail})" if detail else "")
            lines.append(f'        {nid}["{label}"]')
        lines.append("    end")
    for e in edges:
        lines.append(f'    {e["source"]} -- "{e["label"]}" --> {e["target"]}')
    for idx, e in enumerate(edges):
        style = FLOW_STYLES[e["kind"]]
        dash = ",stroke-dasharray:4 3" if e["kind"] == "foreign" else ""
        lines.append(
            f"    linkStyle {idx} stroke:{style['color']},"
            f"stroke-width:2px{dash}"
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


def structure_to_dot(
    p, industry_names=None, good_names=None, io_threshold=0.01
):
    """
    The structure as Graphviz DOT text. Graphviz ranks nodes and routes
    edges far better than Mermaid, so this is the renderer to reach for once
    a model has many industries. Emitting the text needs no extra
    dependency; rendering it needs the Graphviz `dot` program.

    Args:
        p (OG-Core Specifications object): model parameters
        industry_names (list): labels for the M industries
        good_names (list): labels for the I consumption goods
        io_threshold (scalar): omit coefficients below this share

    Returns:
        (str): DOT source, renderable with `dot -Tsvg`
    """
    nodes, edges = get_structure(p, industry_names, good_names, io_threshold)
    lines = [
        "digraph structure {",
        "    rankdir=LR;",
        "    splines=spline;",
        "    bgcolor=transparent;",
        '    node [shape=box, style="rounded,filled", '
        'fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=8, penwidth=1.4];',
    ]
    for group, caption in _MERMAID_GROUPS:
        members = [n for n, v in nodes.items() if v["group"] == group]
        if not members:
            continue
        s = NODE_STYLES[group]
        lines.append(f"    subgraph cluster_{group} {{")
        lines.append(
            f'        label="{caption}"; color="#B4B2A9"; fontsize=9;'
        )
        for nid in members:
            detail = nodes[nid]["detail"]
            label = nodes[nid]["label"] + (f"\\n{detail}" if detail else "")
            lines.append(
                f'        {nid} [label="{label}", fillcolor="{s["fc"]}", '
                f'color="{s["ec"]}", fontcolor="{s["tc"]}"];'
            )
        lines.append("    }")
    for e in edges:
        style = FLOW_STYLES[e["kind"]]
        dashed = ', style="dashed"' if e["kind"] == "foreign" else ""
        lines.append(
            f'    {e["source"]} -> {e["target"]} [label="{e["label"]}", '
            f'color="{style["color"]}"{dashed}];'
        )
    lines.append("}")
    return "\n".join(lines)
