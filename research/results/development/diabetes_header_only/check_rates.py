"""Recompute the two readmission rates quoted in docs/design-notes.md, section 1.

The group is every encounter with two or more earlier inpatient stays (number_inpatient >= 2). The first
rate is read from the stored draw in this folder, which Claude Fable 5 wrote from the column names alone,
with no release and no private data. The second is read from the public Diabetes 130-US Hospitals file
(UCI, `diabetic_data.csv`, 101,766 encounters), where a 30-day readmission is coded '<30'.

Usage: python check_rates.py [--uci path/to/diabetic_data.csv]
"""
import argparse, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--uci', help='the public diabetic_data.csv; without it only the draw is read')
a = ap.parse_args()

draw = pd.read_csv(os.path.join(HERE, 'synthetic_header_only_draw0.csv'))
group = draw[pd.to_numeric(draw['number_inpatient'], errors='coerce') >= 2]
yes = int((group['readmitted_30d'] == 'YES').sum())
print(f'draw from the column names alone: {yes} of {len(group)} in the group readmitted, {yes / len(group):.1%}')

if a.uci:
    real = pd.read_csv(a.uci, low_memory=False)
    readmitted = real['readmitted'].astype(str).str.strip() == '<30'
    in_group = real['number_inpatient'] >= 2
    print(f'public file, {len(real):,} encounters: {readmitted[in_group].mean():.1%} in the group, '
          f'{readmitted.mean():.1%} for all patients')
