from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

output = Path('generated')
previous_trainings = list(map(str, output.glob('cycle_*/selection.csv')))
last = max(int(re.findall("[\d]+", cycle)[0]) for cycle in previous_trainings)

means = []
for i in range(last+1):
    selection = pd.read_csv(f"generated/cycle_{i:04d}/selection.csv")
    print(f'Cycle {i:4d}. Mean: {selection.cnnaffinity.mean():4f}, SD: {selection.cnnaffinity.std():4f}, Min: {selection.cnnaffinity.min():4f}, Max: {selection.cnnaffinity.max():4f}, '
          f'Below -4: {sum(selection.cnnaffinity < -4):3d}, Below -5: {sum(selection.cnnaffinity < -5):3d}, Below -6: {sum(selection.cnnaffinity < -6):3d}, ')
    means.append(selection.cnnaffinity.mean())

print("Linear regression: ", linregress(range(last+1), means))
print('hi')
