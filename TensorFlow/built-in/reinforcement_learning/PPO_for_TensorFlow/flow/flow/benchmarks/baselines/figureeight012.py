# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evaluates the baseline performance of figureeight without RL control.

Baseline is human acceleration and intersection behavior.
"""

import numpy as np
from flow.core.experiment import Experiment
from flow.core.params import SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController
from flow.controllers import ContinuousRouter
from flow.benchmarks.figureeight0 import flow_params


def figure_eight_baseline(num_runs, render=True):
    """Run script for all figure eight baselines.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render : bool, optional
            specifies whether to use the gui during execution

    Returns
    -------
        Experiment
            class needed to run simulations
    """
    sim_params = flow_params['sim']
    env_params = flow_params['env']

    # modify the rendering to match what is requested
    sim_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    # we want no autonomous vehicles in the simulation
    vehicles = VehicleParams()
    vehicles.add(veh_id='human',
                 acceleration_controller=(IDMController, {'noise': 0.2}),
                 routing_controller=(ContinuousRouter, {}),
                 car_following_params=SumoCarFollowingParams(
                     speed_mode='obey_safe_speed',
                 ),
                 num_vehicles=14)

    flow_params['env'].horizon = env_params.horizon
    exp = Experiment(flow_params)

    results = exp.run(num_runs)
    avg_speed = np.mean(results['returns'])

    return avg_speed


if __name__ == '__main__':
    runs = 2  # number of simulations to average over
    res = figure_eight_baseline(num_runs=runs)

    print('---------')
    print('The average speed across {} runs is {}'.format(runs, res))
