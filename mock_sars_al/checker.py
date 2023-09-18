from pathlib import Path
import re

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

output = Path('generated')
previous_trainings = list(map(str, output.glob('cycle_*/selection.csv')))
last = max(int(re.findall("[\d]+", cycle)[0]) for cycle in previous_trainings)

oracle = pd.read_csv('negative_oracle.csv')
oracle.rename({'cnnaffinity': 'oracle'}, axis=1, inplace=True)
init = pd.read_csv('random_starter.csv')
init.rename({'cnnaffinity': 'c0'}, axis=1, inplace=True)

means = []
for_pd = [oracle.oracle, init.c0]
for i in range(1, last+1):
    selection = pd.read_csv(f"{output}/cycle_{i:04d}/selection.csv")
    print(f'Cycle {i:4d}. Mean: {selection.cnnaffinity.mean():4.2f}, SD: {selection.cnnaffinity.std():4.2f}, Min: {selection.cnnaffinity.min():4.2f}, Max: {selection.cnnaffinity.max():4.2f}, '
          f'Below -4: {sum(selection.cnnaffinity < -4):3d}, Below -5: {sum(selection.cnnaffinity < -5):3d}, Below -6: {sum(selection.cnnaffinity < -6):3d}, ')
    means.append(selection.cnnaffinity.mean())
    newcol = f'c{i+1}'
    newdf = selection.rename({'cnnaffinity': newcol}, axis=1)
    for_pd.append(newdf[newcol])

g = sns.catplot(data=pd.concat(for_pd, axis=1), kind="violin")
g.set(ylabel="CNNAffinity")
plt.savefig('violin.png')

# print("Linear regression: ", linregress(range(last+1), means))
