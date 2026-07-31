# Structure and calibration visuals

`generate_gallery.py` produces the visuals in `ogcore.structure_plots` for a
country model. Everything it draws except the calibration fit comes from a
`Specifications` object alone, so a calibration can be inspected before it has
ever been solved.

The figures here were generated from the OG-PHL multi-industry calibration
(M = 8 industries, I = 5 consumption goods) with:

```
cd ~/Projects/OG-PHL
PYTHONPATH=~/Projects/OG-Core .venv/bin/python \
    ~/Projects/OG-Core/examples/structure_plots/generate_gallery.py \
    --package ogphl \
    --params ogphl_default_parameters.json \
    --overlay ogphl_multisector_default_parameters.json \
    --moments ~/Projects/OG-Core/examples/structure_plots/phl_moments.csv \
    --out ~/Projects/OG-Core/examples/structure_plots --prefix phl_
```

Swap `ogphl` for `ogzaf`, `ogidn`, `ogbra` or `ogeth`, and drop `--overlay` for
a single-industry model. Each country's data stays in its own repository; the
script only reads it.

| File | What it shows |
| --- | --- |
| `phl_circular_flow.png` | The institutional linkages. Which agents exist, which flows connect them, and which tax, transfer and cross-border channels the parameterization switches on. The chips in each box carry one parameter each, tinted by whether the country calibrated it. |
| `phl_io_heatmap.png` | The input-output bridge matrix: how much of each industry's output composes each consumption good. |
| `phl_calibration_status.png` | Every structural parameter, coloured by whether its value differs from the one OG-Core ships. Spreading OG-Core's value across new industries counts as inherited, not calibrated. |
| `phl_calibration_fit.png` | How close the solved model lands to the moments it targets, from `phl_moments.csv`. |
| `phl_mermaid.png` | The same structure as a graph, rendered from Mermaid source. |

## The moment table

`--moments` takes a CSV with `name`, `model` and `target`, and optionally
`group` and `source`. These are the only numbers the visuals cannot read from
the parameters: they come from a solved model and from published sources.
`phl_moments.csv` is the table reported in OG-PHL's remittances and revenue
calibration.

## The Mermaid graph

The graph is written as a PNG by default, which needs the Mermaid CLI:

```
npm install -g @mermaid-js/mermaid-cli
```

That is the one piece of this example that is not pure Python. Rendering
locally also means the ELK layout is available, so the graph gets orthogonal
edge routing, which is much easier to follow than the default engine's long
curves.

Pass `--mermaid-format mmd` to write the Mermaid source instead. That needs
nothing installed and, carrying no layout directive, renders in GitHub
markdown and in Jupyter Book.
