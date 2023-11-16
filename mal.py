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
import warnings

import pandas as pd
import numpy
import requests
from pathlib import Path

from rdkit import Chem

from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import ALCycler


class ActiveLearner:
    def __init__(self, config, initial_values=pd.DataFrame()):
        self.feature = config.model_config.targets.params.feature_column
        generated = Path('generated')

        previous_trainings = list(map(str, generated.glob('cycle_*/selection.csv')))
        if config.training_pool != '':
            previous_trainings += [config.training_pool]
        config.training_pool = ','.join(previous_trainings)
        print(f'Detected {len(config.training_pool)} cycles. ')

        if previous_trainings:
            # use the latest full .csv which already has Training set
            # instead of the initial one
            config.virtual_library = str(generated / f"cycle_{len(previous_trainings):04d}" / "virtual_library_with_predictions.csv")

        self.cycle = max(len(previous_trainings), 0)
        self.cycler = ALCycler(config)
        self.virtual_library = self.cycler.get_virtual_library()
        print(f'Launching summary: Feature: {len(self.virtual_library[self.virtual_library[self.feature].notna()])}, '
              f'Training: {len(self.virtual_library[self.virtual_library.Training == True])}, '
              f'Enamines: {len(self.virtual_library[self.virtual_library.enamine_id.notna()])}, '
              f'Enamines Training: {len(self.virtual_library[self.virtual_library.enamine_id.notna() & self.virtual_library.Training == True])}')

    def report(self):
        # select only the ones that have been chosen before
        best_finds = self.virtual_library[self.virtual_library[self.feature] < -6]  # -6 is about 5% of the best cases
        print(f"IT: {self.cycle}, lib: {len(self.virtual_library)}, "
              f"training: {len(self.virtual_library[self.virtual_library.Training])}, "
              f"feature no: {len(self.virtual_library[~self.virtual_library[self.feature].isna()])}, "
              f"<-6 feature: {len(best_finds)}")

    def get_next_best(self, force_random=False):
        self.cycle += 1

        # pick random molecules
        rows_not_yet_computed = self.virtual_library[~self.virtual_library[self.feature].notnull()]
        if len(rows_not_yet_computed) == len(self.virtual_library) or force_random:
            print("Selecting random molecules to study. ")
            chosen_ones = rows_not_yet_computed.sample(self.cycler._cycle_config.selection_config.num_elements)
        else:
            start_time = time.time()
            chosen_ones, virtual_library_regression = self.cycler.run_cycle(self.virtual_library)
            print(f"AL: generated next best {len(chosen_ones)} in: {time.time() - start_time:.1f}s")

        enamines = self.virtual_library[self.virtual_library.enamine_id.notna() &
                                              self.virtual_library[self.feature].isna()]
        if len(enamines) > 0:
            print(f"Adding on top {len(enamines)} Enamine molecules to be computed.")

        return pd.concat([chosen_ones, enamines])

    def set_feature_result(self, smiles, value):
        self.virtual_library.loc[self.virtual_library.Smiles == smiles,
                                 [self.feature, ncl_cycle.TRAINING_KEY]] = value, True

    def add_enamine_molecules(self, scaffold, limit=300):
        """
        For the best scoring molecules that were grown,
        check if there are similar molecules in Enamine REAL database,
        if they are, add them to the dataset.

        @number: The maximum number of molecules that
            should be added from the Enamine REAL database.
        @similarity_cutoff: Molecules queried from the database have to be
            at least this similar to the original query.
        @scaffold: The scaffold molecule that has to be present in the search.
            If None, the requirement will be ignored.
        """

        # get the best performing molecules
        vl = self.virtual_library.sort_values(by='cnnaffinity')
        best_vl_for_searching = vl[:100]

        # nothing to search for yet
        if len(best_vl_for_searching[~best_vl_for_searching.cnnaffinity.isna()]) == 0:
            return

        if len(set(best_vl_for_searching.h)) > 1:
            raise NotImplementedError('Multiple growth vectors are used. ')

        # filter out previously queried molecules
        new_searches = best_vl_for_searching[best_vl_for_searching.enamine_searched == False]
        smiles_to_search = list(new_searches.Smiles)

        start = time.time()
        print('Querying Enamine REAL. ')
        from smallworld_api import SmallWorld
        sw = SmallWorld()
        try:
            results: pd.DataFrame = sw.search_many(smiles_to_search, dist=5, db=sw.REAL_dataset, length=5000)
        except requests.exceptions.HTTPError as HTTPError:
            print("Enamine API call failed. ", HTTPError)
            return
        print(f"Enamine returned with {len(results)} rows in {time.time() - start:.2f}s.")

        # check if they have the necessary core group
        with_scaffold = []
        for i, row in results.iterrows():
            try:
                mol = Chem.MolFromSmiles(row.hitSmiles)
                if mol.HasSubstructMatch(scaffold):
                    with_scaffold.append(True)
                else:
                    with_scaffold.append(False)
            except Exception as E:
                warnings.warn("Checking if the Enamine molecule has the scaffold failed: " + str(E))
                with_scaffold.append(False)

        print(f"Tested scaffold presence. Kept {sum(with_scaffold)}/{len(with_scaffold)}.")

        if len(with_scaffold) > 0:
            similar = results[with_scaffold]
        else:
            similar = pd.DataFrame(columns=results.columns)

        similar.sort_values(by='ecfp4', inplace=True, ascending=False)

        # select the top 300 cases
        top_similar = similar[:limit]

        # filter out Enamine molecules which were previously added
        new_enamines = top_similar[~top_similar.id.isin(vl.enamine_id)]

        # fixme: automate creating empty dataframes. Allow to configure default values initially.
        new_df = pd.DataFrame({'Smiles': new_enamines.hitSmiles,
                      self.feature: numpy.nan,
                      'h': vl.h[0], # fixme: for now assume that only one vector is used
                      'enamine_id': new_enamines.id,
                      'enamine_searched': False,
                      'Training': False })

        print("Adding: ", len(new_df))
        self.virtual_library = pd.concat([vl, new_df], ignore_index=True)


    def csv_cycle_summary(self, chosen_ones):
        cycle_dir = Path(f"generated/cycle_{self.cycle:04d}")
        print(f"Next generation report saved: {cycle_dir}")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        self.virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
        chosen_ones.to_csv(cycle_dir / "selection.csv", columns=self.cycler._cycle_config.selection_config.selection_columns, index=False)
        self.report()
