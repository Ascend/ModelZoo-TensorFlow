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
"""Contains the base lane change controller class."""


class BaseLaneChangeController:
    """Base class for lane-changing controllers.

    Instantiates a controller and forces the user to pass a
    lane_changing duration to the controller.

    Usage
    -----
    >>> from flow.core.params import VehicleParams
    >>> from flow.controllers import SimLaneChangeController
    >>> vehicles = VehicleParams()
    >>> vehicles.add("human",
    >>>              lane_change_controller=(SimLaneChangeController, {}))

    Note: You can replace SimLaneChangeController with any lane changing
    controller of your choice.

    Parameters
    ----------
    veh_id : str
        ID of the vehicle this controller is used for
    lane_change_params : dict
        Dictionary of lane changes params that may optional contain
        "min_gap", which denotes the minimize safe gap (in meters) a car
        is willing to lane-change into.
    """

    def __init__(self, veh_id, lane_change_params=None):
        """Instantiate the base class for lane-changing controllers."""
        if lane_change_params is None:
            lane_change_params = {}

        self.veh_id = veh_id
        self.lane_change_params = lane_change_params

    def get_lane_change_action(self, env):
        """Specify the lane change action to be performed.

        If discrete lane changes are being performed, the action is a direction

        * -1: lane change right
        * 0: no lane change
        * 1: lane change left

        Parameters
        ----------
        env : flow.envs.Env
            state of the environment at the current time step

        Returns
        -------
        float or int
            requested lane change action
        """
        raise NotImplementedError

    def get_action(self, env):
        """Return the action of the lane change controller.

        Modifies the lane change action to ensure safety, if requested.

        Parameters
        ----------
        env : flow.envs.Env
            state of the environment at the current time step

        Returns
        -------
        float or int
            lane change action
        """
        lc_action = self.get_lane_change_action(env)
        # TODO(ak): add failsafe

        return lc_action
