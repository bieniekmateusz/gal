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
    print(f'Cycle {i:4d}. Mean: {selection.cnnaffinity.mean():4f}, SD: {selection.cnnaffinity.std():4f}, Min: {selection.cnnaffinity.min():4f}, Max: {selection.cnnaffinity.max():4f}')
    means.append(selection.cnnaffinity.mean())
    # plt.hist(selection.cnnaffinity, alpha=0.5)

print("Linear regression: ", linregress(range(last+1), means))
print('hi')
# plt.show()