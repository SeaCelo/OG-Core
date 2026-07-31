import os
import matplotlib
import numpy as np
import pytest

from ogcore import structure_plots
from ogcore.parameters import Specifications

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def default_params():
    return Specifications()


@pytest.fixture(scope="module")
def multi_industry_params():
    """A three-industry, two-good parameterization with an open economy."""
    p = Specifications()
    p.update_specifications(
        {
            "M": 3,
            "I": 2,
            "io_matrix": [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]],
            "alpha_c": [0.4, 0.6],
            "zeta_K": [0.5],
            "zeta_D": [0.3],
        }
    )
    return p


def test_get_structure_wires_io_matrix(multi_industry_params):
    """Every io_matrix coefficient above the threshold becomes an edge."""
    p = multi_industry_params
    nodes, edges = structure_plots.get_structure(p, io_threshold=0.15)
    io = np.atleast_2d(p.io_matrix)

    for m in range(p.M):
        assert f"IND{m}" in nodes
    for i in range(p.I):
        assert f"GOOD{i}" in nodes

    bridge = {
        (e["source"], e["target"])
        for e in edges
        if e["source"].startswith("IND") and e["target"].startswith("GOOD")
    }
    expected = {
        (f"IND{m}", f"GOOD{i}")
        for i in range(p.I)
        for m in range(p.M)
        if io[i, m] > 0.15
    }
    assert bridge == expected


def test_closed_economy_drops_foreign_sector(default_params):
    """With no openness and no remittances there is no rest of the world."""
    p = Specifications()
    p.update_specifications(
        {"zeta_K": [0.0], "zeta_D": [0.0], "alpha_FA": [0.0]}
    )
    nodes, edges = structure_plots.get_structure(p)
    assert "ROW" not in nodes
    assert not [e for e in edges if e["kind"] == "foreign"]

    open_nodes, _ = structure_plots.get_structure(default_params)
    assert "ROW" in open_nodes


def test_inactive_instruments_are_omitted(default_params):
    """A tax only appears when its parameter is non-zero."""
    labels = {
        e["label"]
        for e in structure_plots.get_structure(default_params)[1]
        if e["target"] == "GOV"
    }
    assert "wealth tax" not in labels

    p = Specifications()
    p.update_specifications({"p_wealth": [0.02]})
    labels = {
        e["label"]
        for e in structure_plots.get_structure(p)[1]
        if e["target"] == "GOV"
    }
    assert "wealth tax" in labels


def test_labels_must_match_dimensions(multi_industry_params):
    with pytest.raises(ValueError):
        structure_plots.get_structure(
            multi_industry_params, industry_names=["only one"]
        )


def test_calibration_status_flags_changed_values(default_params):
    """A default parameterization is all defaults; a changed one is not."""
    status, blocks = structure_plots.calibration_status(default_params)
    assert set(status.values()) <= {"default", "missing"}
    assert all(n_cal == 0 for n_cal, _ in blocks.values())

    p = Specifications()
    p.update_specifications({"alpha_T": [0.2]})
    status, blocks = structure_plots.calibration_status(p)
    assert status["alpha_T"] == "calibrated"
    assert blocks["government"][0] == 1


def test_broadcast_across_dimensions_is_not_a_calibration(default_params):
    """
    Spreading OG-Core's own value across new industries is not calibration;
    changing that value is.
    """
    reference = float(np.atleast_1d(default_params.epsilon)[0])

    spread = Specifications()
    spread.update_specifications({"M": 3, "epsilon": [reference] * 3})
    assert (
        structure_plots.calibration_status(spread)[0]["epsilon"] == "default"
    )

    changed = Specifications()
    changed.update_specifications(
        {"M": 3, "epsilon": [reference, reference, reference + 0.4]}
    )
    status = structure_plots.calibration_status(changed)[0]
    assert status["epsilon"] == "calibrated"


@pytest.mark.parametrize(
    "plot_func",
    [
        structure_plots.plot_circular_flow,
        structure_plots.plot_io_heatmap,
        structure_plots.plot_calibration_status,
    ],
    ids=["circular_flow", "io_heatmap", "calibration_status"],
)
def test_plots_return_figures(plot_func, multi_industry_params):
    fig = plot_func(multi_industry_params)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_plots_save_to_path(multi_industry_params, tmp_path):
    out = tmp_path / "structure.png"
    assert (
        structure_plots.plot_circular_flow(
            multi_industry_params, path=str(out)
        )
        is None
    )
    assert out.exists()


def test_mermaid_unbundled_covers_every_node_and_edge(multi_industry_params):
    nodes, edges = structure_plots.get_structure(multi_industry_params)
    text = structure_plots.structure_to_mermaid(
        multi_industry_params, bundle=False
    )
    assert "flowchart LR" in text
    for nid in nodes:
        assert nid in text
    # One linkStyle per edge, so no edge is left with a default color.
    assert text.count("linkStyle ") == len(edges)


def test_bundling_collapses_whole_group_fanouts(multi_industry_params):
    """
    An edge repeated to every industry becomes one edge on the group; an edge
    reaching only some of them stays as it is.
    """
    nodes, edges = structure_plots.get_structure(multi_industry_params)
    bundled = structure_plots.bundle_group_edges(nodes, edges)

    assert len(bundled) < len(edges)
    labor = [
        e for e in bundled if e["label"] == "labor" and e["source"] == "HH"
    ]
    assert len(labor) == 1
    assert labor[0]["target"] == "industry"
    assert labor[0]["bundled"] == multi_industry_params.M

    # Only the last industry supplies investment, so that one is untouched.
    investment = [e for e in bundled if e["label"] == "investment"]
    assert len(investment) == 1
    assert investment[0]["source"] == f"IND{multi_industry_params.M - 1}"

    # Every io coefficient keeps its own edge, because each has its own label.
    io_edges = [
        e
        for e in bundled
        if e["source"].startswith("IND") and e["target"].startswith("GOOD")
    ]
    assert io_edges and all("bundled" not in e for e in io_edges)


def test_collapsing_groups_leaves_one_node_each(multi_industry_params):
    nodes, edges = structure_plots.get_structure(multi_industry_params)
    edges = structure_plots.bundle_group_edges(nodes, edges)
    collapsed, out = structure_plots.collapse_group_nodes(
        nodes,
        edges,
        ("industry", "good"),
        dict(structure_plots._MERMAID_GROUPS),
    )

    assert "industry" in collapsed and "good" in collapsed
    assert not [n for n in collapsed if n.startswith(("IND", "GOOD"))]
    assert f"M = {multi_industry_params.M}" in collapsed["industry"]["detail"]

    # The io coefficients ran between two collapsed groups, so they merge into
    # one edge rather than becoming a self-loop or a list of numbers.
    industry_to_good = [
        e for e in out if e["source"] == "industry" and e["target"] == "good"
    ]
    assert len(industry_to_good) == 1
    assert industry_to_good[0]["label"] == "output"

    # Bequests were already a self-loop and survive as one.
    assert [e for e in out if e["source"] == e["target"] == "HH"]


def test_mermaid_bundled_is_smaller_and_flat(multi_industry_params):
    """The institutional graph needs no subgraphs and far fewer edges."""
    full = structure_plots.structure_to_mermaid(
        multi_industry_params, bundle=False
    )
    bundled = structure_plots.structure_to_mermaid(
        multi_industry_params, bundle=True
    )
    assert bundled.count(" --> ") < full.count(" --> ")
    assert "subgraph" not in bundled
    assert "Production industries" in bundled
    assert bundled.count("linkStyle ") == bundled.count(" --> ")


def test_summarize_structure_reports_off_channels(default_params):
    summary = structure_plots.summarize_structure(default_params)
    assert summary["wealth tax"] is None
    assert "M = 1" in summary["industries"]

    p = Specifications()
    p.update_specifications({"alpha_RM_1": 0.05, "alpha_RM_T": 0.05})
    assert (
        structure_plots.summarize_structure(p)["remittances"] == "5.0% of GDP"
    )


MOMENTS = [
    {"group": "Revenue", "name": "PIT / Y", "model": 0.0312, "target": 0.0312},
    {"group": "Revenue", "name": "CIT / Y", "model": 0.0421, "target": 0.0420},
    {"group": "External", "name": "K_f / K", "model": 0.2027, "target": 0.20},
    {"group": "External", "name": "RM / Y", "model": 0.0541, "target": 0.0812},
]


def test_calibration_fit_bands_by_distance():
    """Each moment lands in the band its distance from target implies."""
    fig = structure_plots.plot_calibration_fit(MOMENTS)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)

    # Exact, 0.24%, 1.35% and -33% off, against tolerances of 1% and 5%.
    deviations = [
        (row["model"] - row["target"]) / abs(row["target"]) for row in MOMENTS
    ]
    assert abs(deviations[0]) < 0.01
    assert abs(deviations[1]) < 0.01
    assert 0.01 < abs(deviations[2]) < 0.05
    assert abs(deviations[3]) > 0.05


def test_calibration_fit_saves_to_path(tmp_path):
    out = tmp_path / "fit.png"
    assert structure_plots.plot_calibration_fit(MOMENTS, path=str(out)) is None
    assert out.exists()


def test_calibration_fit_needs_model_and_target():
    with pytest.raises(ValueError):
        structure_plots.plot_calibration_fit(
            [{"name": "PIT / Y", "model": 0.03}]
        )


def test_calibration_fit_handles_a_zero_target():
    """A zero target falls back to an absolute deviation, without dividing."""
    fig = structure_plots.plot_calibration_fit(
        [{"name": "UBI / Y", "model": 0.004, "target": 0.0}]
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_calibration_fit_handles_an_exact_calibration():
    """Every moment on target still gives a usable axis."""
    exact = [
        {"name": "D / Y", "model": 0.6, "target": 0.6},
        {"name": "K_f / K", "model": 0.2, "target": 0.2},
    ]
    fig = structure_plots.plot_calibration_fit(exact)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_make_visuals_writes_the_whole_set(multi_industry_params, tmp_path):
    out = structure_plots.make_visuals(
        multi_industry_params,
        moments=MOMENTS,
        output_dir=str(tmp_path),
        prefix="phl_",
    )
    assert list(out) == list(structure_plots.VISUALS)
    for name, path in out.items():
        assert os.path.exists(path)
        assert os.path.basename(path).startswith("phl_")
        assert path.endswith(".mmd" if name == "mermaid" else ".png")


def test_make_visuals_returns_objects_without_a_directory(
    multi_industry_params,
):
    out = structure_plots.make_visuals(
        multi_industry_params, ["circular_flow", "mermaid"]
    )
    assert isinstance(out["circular_flow"], matplotlib.figure.Figure)
    assert "flowchart LR" in out["mermaid"]
    matplotlib.pyplot.close(out["circular_flow"])


def test_make_visuals_accepts_one_name(multi_industry_params):
    out = structure_plots.make_visuals(multi_industry_params, "io_heatmap")
    assert list(out) == ["io_heatmap"]
    matplotlib.pyplot.close(out["io_heatmap"])


def test_make_visuals_skips_the_fit_when_no_moments(multi_industry_params):
    """
    Targets come from published sources, so "all" without them means all the
    rest. Asking for the fit by name without them is an error.
    """
    out = structure_plots.make_visuals(multi_industry_params)
    assert "calibration_fit" not in out
    for fig in out.values():
        if isinstance(fig, matplotlib.figure.Figure):
            matplotlib.pyplot.close(fig)

    with pytest.raises(ValueError):
        structure_plots.make_visuals(multi_industry_params, "calibration_fit")


def test_make_visuals_rejects_an_unknown_name(multi_industry_params):
    with pytest.raises(ValueError):
        structure_plots.make_visuals(multi_industry_params, "not_a_visual")


def test_make_visuals_passes_options_through(multi_industry_params):
    out = structure_plots.make_visuals(
        multi_industry_params,
        "mermaid",
        options={"mermaid": {"bundle": False, "link_width": 4}},
    )
    assert "subgraph" in out["mermaid"]
    assert "stroke-width:4px" in out["mermaid"]


def test_mermaid_layout_is_opt_in(multi_industry_params):
    """
    The default carries no layout directive, so the source renders wherever it
    is pasted. GitHub does not load the ELK plugin, so ELK is opt-in.
    """
    default = structure_plots.structure_to_mermaid(multi_industry_params)
    assert not default.startswith("%%{init")
    assert default.startswith("flowchart LR")

    elk = structure_plots.structure_to_mermaid(
        multi_industry_params, layout="elk"
    )
    assert elk.startswith('%%{init: {"layout": "elk"}}%%')
    assert "flowchart LR" in elk


def test_render_mermaid_says_what_is_missing(multi_industry_params, tmp_path):
    """Without the CLI the error names the install, not an odd failure."""
    source = structure_plots.structure_to_mermaid(multi_industry_params)
    with pytest.raises(RuntimeError, match="mermaid-cli"):
        structure_plots.render_mermaid(
            source,
            str(tmp_path / "graph.png"),
            command=["definitely-not-a-real-mermaid-cli"],
        )


def test_make_visuals_honors_the_figure_format(
    multi_industry_params, tmp_path
):
    out = structure_plots.make_visuals(
        multi_industry_params,
        ["circular_flow", "io_heatmap"],
        output_dir=str(tmp_path),
        fmt="svg",
    )
    for path in out.values():
        assert path.endswith(".svg")
        assert os.path.exists(path)
