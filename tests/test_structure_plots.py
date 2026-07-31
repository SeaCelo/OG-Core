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


def test_calibration_status_honors_explicit_names(default_params):
    """A parameter named by the caller counts even at the default value."""
    status, _ = structure_plots.calibration_status(
        default_params, calibrated_params=[{"sigma": 1.5}]
    )
    assert status["sigma"] == "calibrated"
    assert status["frisch"] == "default"


@pytest.mark.parametrize(
    "plot_func",
    [
        structure_plots.plot_circular_flow,
        structure_plots.plot_io_bridge,
        structure_plots.plot_io_heatmap,
        structure_plots.plot_calibration_status,
    ],
    ids=["circular_flow", "io_bridge", "io_heatmap", "calibration_status"],
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


def test_mermaid_covers_every_node_and_edge(multi_industry_params):
    nodes, edges = structure_plots.get_structure(multi_industry_params)
    text = structure_plots.structure_to_mermaid(multi_industry_params)
    assert text.startswith("flowchart LR")
    for nid in nodes:
        assert nid in text
    # One linkStyle per edge, so no edge is left with a default color.
    assert text.count("linkStyle ") == len(edges)


def test_dot_is_balanced(multi_industry_params):
    nodes, edges = structure_plots.get_structure(multi_industry_params)
    text = structure_plots.structure_to_dot(multi_industry_params)
    assert text.startswith("digraph structure {")
    assert text.rstrip().endswith("}")
    assert text.count("{") == text.count("}")
    assert text.count(" -> ") == len(edges)


def test_summarize_structure_reports_off_channels(default_params):
    summary = structure_plots.summarize_structure(default_params)
    assert summary["wealth tax"] is None
    assert "M = 1" in summary["industries"]

    p = Specifications()
    p.update_specifications({"alpha_RM_1": 0.05, "alpha_RM_T": 0.05})
    assert (
        structure_plots.summarize_structure(p)["remittances"] == "5.0% of GDP"
    )
