# The unconditioned readmission rate (a development measurement)

These files support one number in `docs/design-notes.md`, section 1: an unconditioned language model put the
30-day readmission rate of frequently admitted diabetic patients at 98.5%, where the real rate is 21.4%.

| file | what it is |
|---|---|
| `synthetic_header_only_draw0.csv` | 300 encounters written by Claude Fable 5 (`claude-fable-5`, Anthropic API) from the column names alone, with no release and no private data, on 4 September 2026 during development, before the paper's experiments. It holds no real record. |
| `check_rates.py` | Recomputes both rates: the draw's rate in the group, and the public file's rate in the same group and overall. |

The group is every encounter whose patient had two or more inpatient stays in the year before it (`number_inpatient`
of 2 or more). In the draw, 65 of the 66 encounters in that group are readmitted within 30 days (98.5%). In the public
Diabetes 130-US Hospitals file from the UCI repository (`diabetic_data.csv`, 101,766 encounters), the same group's
rate is 21.4%, against 11.2% for all patients.

```
python check_rates.py --uci path/to/diabetic_data.csv
```

This was a single development draw with the prompts of that time, not one of the paper's configurations. It is
kept because it shows the failure the release exists to prevent: a model that has seen no data can only repeat
its own assumptions, and here its assumption is wrong by 77 percentage points.
