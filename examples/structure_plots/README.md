# Structure and calibration visuals

`generate_gallery.py` produces the visuals in `ogcore.structure_plots` for a
country model. Everything it draws except the calibration fit comes from a
`Specifications` object alone, so a calibration can be inspected before it has
ever been solved.

The figures here were generated from the OG-PHL multi-industry calibration
(M = 8 industries, I = 5 consumption goods) with:

```
cd ~/Projects/OG-PHL
.venv/bin/python \
    <this-checkout>/examples/structure_plots/generate_gallery.py \
    --package ogphl \
    --params ogphl_default_parameters.json \
    --overlay ogphl_multisector_default_parameters.json \
    --moments <this-checkout>/examples/structure_plots/phl_moments.csv \
    --out <this-checkout>/examples/structure_plots --prefix phl_
```

Swap `ogphl` for `ogzaf`, `ogidn`, `ogbra` or `ogeth`, and drop `--overlay` for
a single-industry model. Each country's data stays in its own repository; the
script only reads it.

Note what the command does **not** do: it sets no `PYTHONPATH` and installs
nothing. It runs in the country's own environment, so OG-Core is whichever
build that model is pinned to, and the script loads `structure_plots` by path
from the checkout it ships in. That combination is deliberate, and it is the
only one that works. A country calibration frequently needs an OG-Core newer
than the one carrying these visuals -- OG-PHL's parameters require
`initial_wealth_ratio`, and an OG-Core without it refuses the file outright --
while the build that has it does not carry this module. Shadowing OG-Core with
`PYTHONPATH`, or installing this branch into the country's environment, breaks
one half or the other. Neither is necessary.

## Read the baseline before reading the calibration figures

`calibration_status` calls a parameter calibrated when its value differs from
the one OG-Core ships, so its verdict is relative to the build that supplied
the comparison, and both calibration figures print that build on themselves.
This is not a footnote. The same Philippine parameters read 7 of 12 calibrated
in the government block against one OG-Core build and 10 of 12 against another.
Take the figure and the baseline together, or not at all.

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
