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
import os.path
import time
import pandas as pd
from pathlib import Path

from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import ALCycler


class ActiveLearner:
    def __init__(self, config, initial_values=pd.DataFrame()):
        generated = Path('generated')

        previous_trainings = list(map(str, generated.glob('cycle_*/selection.csv')))
        if config.training_pool != '':
            previous_trainings += [config.training_pool]
        config.training_pool = ','.join(previous_trainings)
        print('Using trainings: ', config.training_pool)

        if previous_trainings:
            # use the latest full .csv which already has Training set
            # instead of the initial one
            config.virtual_library = str(generated / f"cycle_{len(previous_trainings)-1:04d}" / "virtual_library_with_predictions.csv")

        self.cycle = max(len(previous_trainings) - 1, 0)
        self.cycler = ALCycler(config)
        self.virtual_library = self.cycler.get_virtual_library()
        print(f'Feature: {len(self.virtual_library[self.virtual_library.cnnaffinity.notna()])}, '
              f'Training: {len(self.virtual_library[self.virtual_library.Training == True])}, '
              f'Enamines: {len(self.virtual_library[self.virtual_library.enamine_id.notna()])}, '
              f'Enamines Training: {len(self.virtual_library[self.virtual_library.enamine_id.notna() & self.virtual_library.Training == True])}')

    def report(self):
        # select only the ones that have been chosen before
        best_finds = self.virtual_library[self.virtual_library.cnnaffinity < -6]  # -6 is about 5% of the best cases
        print(f"IT: {self.cycle}, lib: {len(self.virtual_library)}, "
              f"training: {len(self.virtual_library[self.virtual_library.Training])}, "
              f"cnnaffinity no: {len(self.virtual_library[~self.virtual_library.cnnaffinity.isna()])}, "
              f"<-6 cnnaff: {len(best_finds)}")

    def get_next_best(self):
        # in the first iteration there is no data, pick random molecules
        not_null_rows = self.virtual_library[self.virtual_library.cnnaffinity.notnull()]
        if len(not_null_rows) == 0:
            # there is nothing dedicated to Training yet
            assert len(self.virtual_library[self.virtual_library.Training == True]) == 0

            random_starter = self.virtual_library.sample(self.cycler._cycle_config.selection_config.num_elements)
            return random_starter

        start_time = time.time()
        chosen_ones, virtual_library_regression = self.cycler.run_cycle(self.virtual_library)
        print(f"Found next best {len(chosen_ones)} in: {time.time() - start_time:.1f}")

        enamines = virtual_library_regression[virtual_library_regression.enamine_id.notna() &
                                              virtual_library_regression.cnnaffinity.isna()]
        if len(enamines) > 0:
            print(f"Adding on top {len(enamines)} Enamine molecules to be computed.")
        self.cycle += 1
        return pd.concat([chosen_ones, enamines])

    def set_answer(self, smiles, result):
        # add this result
        self.virtual_library.loc[self.virtual_library.Smiles == smiles,
                                 ['cnnaffinity', ncl_cycle.TRAINING_KEY]] = result['cnnaffinity'], True

    def csv_cycle_summary(self, chosen_ones):
        cycle_dir = Path(f"generated/cycle_{self.cycle:04d}")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        self.virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
        chosen_ones.to_csv(cycle_dir / "selection.csv", columns=self.cycler._cycle_config.selection_config.selection_columns, index=False)
        self.report()


if os.path.exists("negative_oracle.csv"):
    oracle = pd.read_csv("negative_oracle.csv")
def compute_fegrow(smiles):
    result = oracle[oracle.Smiles == smiles]
    return {'cnnaffinity': result.cnnaffinity.values[0]}


def expand_chemical_space(al):
    """
    For now, add up to a 100 of new random smiles as datapoints.
    """
    extras = oracle.sample(100).drop(columns=['cnnaffinity'], axis=1)
    not_yet_in = extras[~extras.Smiles.isin(al.virtual_library.Smiles.values)]
    not_yet_in = not_yet_in.assign(Training=False)   # fixme: this will break if we use a different keyword
    print(f'Adding {len(not_yet_in)} smiles out of 100')

    extended = pd.concat([al.virtual_library, not_yet_in], ignore_index=True)
    al.virtual_library = extended


if __name__ == '__main__':
    config = get_gaussian_process_config()
    config.virtual_library = "chemical_space_smiles_500.csv"
    config.selection_config.num_elements = 30  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles"]
    config.model_config.targets.params.feature_column = 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048

    al = ActiveLearner(config)

    for i in range(5):
        chosen_ones = al.get_next_best()
        for i, row in chosen_ones.iterrows():
            result = compute_fegrow(row.Smiles)  # TODO no conformers? penalise
            al.set_answer(row.Smiles, result)
            # update for record keeping
            chosen_ones.at[i, 'cnnaffinity'] = result['cnnaffinity']
        al.csv_cycle_summary(chosen_ones)
        expand_chemical_space(al)

    print('hi')

