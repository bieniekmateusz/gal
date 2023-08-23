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

import glob
import pandas as pd
from pathlib import Path

from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import MatCycle


if __name__ == '__main__':
    config = get_gaussian_process_config()
    config.training_pool = "basic_loop/initial_training_set.csv"
    config.virtual_library = "basic_loop/virtual_library.csv"
    config.selection_config.num_elements = 3    # how many new to select

    AL = MatCycle(config)
    virtual_library = AL.get_virtual_library()

    for cycle_id in range(2):
        print(f"Lib size: {len(virtual_library)} with training size: {len(virtual_library[virtual_library.Training])}")
        selections, virtual_library_regression = AL.run_cycle(virtual_library)

        # the new selections are now also part of the training set
        virtual_library_regression.loc[selections.index, ncl_cycle.TRAINING_KEY] = True

        # TODO no conformers? penalise

        # expand the virtual library
        new_record = pd.DataFrame([{'Smiles': "CN(C(=O)c1cn(C)nc1-c1ccc(F)cc1F)c1nc2ccccc2n1C", ncl_cycle.TRAINING_KEY: False}])
        expanded_library = pd.concat([virtual_library_regression, new_record], ignore_index=True)
        virtual_library = expanded_library

        cycle_dir = Path(f"cycle_{cycle_id}")
        cycle_dir.mkdir(exist_ok=True)
        virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
        selections.to_csv(cycle_dir / "selection.csv", columns=config.selection_config.selection_columns, index=False)
