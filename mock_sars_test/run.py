# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for running a single cycle of active learning."""
import time
import re
import pandas as pd
import numpy as np
from pathlib import Path

from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import ALCycler

oracle = pd.read_csv("oracle.csv")
oracle.sort_values(by='cnnaffinity', ascending=False, inplace=True)
# find 5% best values cutoff
cutoff = oracle[:int(0.05*len(oracle))].cnnaffinity.values[0]

def ask_oracle(chosen_ones, virtual_library):
    # check and return all the values for the smiles
    # look up and overwrite the values in place

    # look the up by smiles
    oracle_has_spoken = chosen_ones.merge(oracle, on=['Smiles'])
    # get the correct affinities
    chosen_ones.cnnaffinity = -oracle_has_spoken.cnnaffinity_y.values
    assert np.all(chosen_ones.Smiles.values == oracle_has_spoken.Smiles.values)
    # update the main dataframe
    virtual_library.update(chosen_ones)

def report(virtual_library, start_time):
    # select only the ones that have been chosen before
    best_finds = virtual_library[virtual_library.cnnaffinity < -6]  #-6 is about 5% of the best cases
    print(f"IT: {cycle_id},Lib size: {len(virtual_library)}, "
          f"training size: {len(virtual_library[virtual_library.Training])}, "
          f"cnnaffinity 0: {len(virtual_library[virtual_library.cnnaffinity == 0])}, "
          f"<-6 cnnaff: {len(best_finds)}, "
          f"time: {time.time() - start_time}")

if __name__ == '__main__':
    output = Path('generated')
    previous_trainings = list(map(str, output.glob('cycle_*/selection.csv')))
    print('Attaching trainings:', previous_trainings)

    config = get_gaussian_process_config()
    config.training_pool = ','.join(["init.csv"] + previous_trainings)
    config.virtual_library = "large.csv"
    config.selection_config.num_elements = 100    # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles"]
    config.model_config.targets.params.feature_column = 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048

    AL = ALCycler(config)
    virtual_library = AL.get_virtual_library()

    cycle_start = 0
    if previous_trainings:
        cycle_start = max(int(re.findall("[\d]+", cycle)[0]) for cycle in previous_trainings)

    for cycle_id in range(cycle_start, 400):
        start_time = time.time()
        chosen_ones, virtual_library_regression = AL.run_cycle(virtual_library)

        # the new selections are now also part of the training set
        virtual_library_regression.loc[chosen_ones.index, ncl_cycle.TRAINING_KEY] = True
        ask_oracle(chosen_ones, virtual_library_regression)  # TODO no conformers? penalise
        virtual_library = virtual_library_regression

        # expand the virtual library
        # if len(virtual_library[virtual_library.Smiles == "CN(C(=O)c1cn(C)nc1-c1ccc(F)cc1F)c1nc2ccccc2n1C"]) == 0:
        #     new_record = pd.DataFrame([{'Smiles': "CN(C(=O)c1cn(C)nc1-c1ccc(F)cc1F)c1nc2ccccc2n1C", ncl_cycle.TRAINING_KEY: False}])
        #     expanded_library = pd.concat([virtual_library_regression, new_record], ignore_index=True)
        #     virtual_library = expanded_library

        cycle_dir = Path(f"generated/cycle_{cycle_id:04d}")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
        chosen_ones.to_csv(cycle_dir / "selection.csv", columns=config.selection_config.selection_columns, index=False)

        report(virtual_library, start_time)
