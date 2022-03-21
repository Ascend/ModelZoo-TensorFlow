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
"""Contains the Bay Bridge network class."""

from flow.networks.base import Network

# Use this to ensure that vehicles are only placed in the edges of the Bay
# Bridge moving from Oakland to San Francisco.
EDGES_DISTRIBUTION = [
    '236348360#1',
    '157598960',
    '11415208',
    '236348361',
    '11198599',
    '11198595.0',
    '11198595.656.0',
    '340686911#3',
    '23874736',
    '119057701',
    '517934789',
    '236348364',
    '124952171',
    'gneE0',
    '11198599',
    '124952182.0',
    '236348360#0',
    '497579295',
    '340686911#2.0.0',
    '340686911#1',
    '394443191',
    '322962944',
    '32661309#1.0',
    '90077193#1.777',
    '90077193#1.0',
    '90077193#1.812',
    'gneE1',
    '32661316',
    '4757680',
    '124952179',
    '119058993',
    '28413679',
    '11197898',
    '123741311',
    '123741303',
    '90077193#0',
    '28413687#1',
    '11197889',
    '123741382#0',
    '123741382#1',
    'gneE3',
    '340686911#0.54.0',
    '340686911#0.54.54.0',
    '340686911#0.54.54.127.0',
    '340686911#2.35',
]


class BayBridgeNetwork(Network):
    """A network used to simulate the Bay Bridge.

    The bay bridge was originally imported from OpenStreetMap and subsequently
    modified to more closely match the network geometry of the actual Bay
    Bridge. Vehicles are only allowed to exist of and traverse the edges
    leading up to and which the westbound Bay Bridge.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import BayBridgeNetwork
    >>>
    >>> network = BayBridgeNetwork(
    >>>     name='bay_bridge',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams()
    >>> )
    """

    def specify_routes(self, net_params):
        """See parent class.

        Routes for vehicles moving through the bay bridge from Oakland to San
        Francisco.
        """
        rts = {
            '11198593': ['11198593', '11198595.0'],
            '157598960': ['157598960', '11198595.0'],
            '11198595.0': ['11198595.0', '11198595.656.0'],
            '11198595.656.0': ['11198595.656.0', 'gneE5'],
            'gneE5': ['gneE5', '340686911#2.0.13'],
            '124952171': ['124952171', '11198599'],
            '340686911#1': ['340686911#1', '340686911#2.0.0'],
            '340686911#2.0.0': ['340686911#2.0.0', '340686911#2.0.13'],
            '340686911#2.0.13': ['340686911#2.0.13', '340686911#2.35'],
            '340686911#0.54.54.127.74':
            ['340686911#0.54.54.127.74', '340686911#1'],
            '340686911#3': ['340686911#3', '236348361'],
            '236348361': ['236348361', '236348360#0'],
            '236348360#0': ['236348360#0', '236348360#1'],
            '236348360#0_1': ['236348360#0', '322962944'],
            '236348360#1': ['236348360#1', '517934789'],
            '517934789': ['517934789', '236348364'],
            '236348364': ['236348364', '497579295'],
            '35536683': ['35536683', '497579295'],
            '497579295': ['497579295', '11415208'],
            '11415208': ['11415208', '23874736'],
            '119057701': ['119057701', '394443191'],
            '23874736': ['23874736', '394443191'],
            '183343422': ['183343422', '32661316'],
            '183343422_1': ['183343422', '4757680'],
            '393649534': ['393649534', '124952179'],
            '32661316': ['32661316', '124952179'],
            '124952179': ['124952179', '157598960'],
            '124952179_1': ['124952179', '124952171'],
            '4757680': ['4757680', '32661309#0'],
            '11189946': ['11189946', '119058993'],
            '119058993': ['119058993', '28413679'],
            '28413679': ['28413679', '11197898'],
            '11197898': ['11197898', '123741311'],
            '123741311': ['123741311', '123741303'],
            '123741303': ['123741303', '90077193#0'],
            '28413687#0': ['28413687#0', '28413687#1'],
            '28413687#1': ['28413687#1', '123741382#0'],
            '11197889': ['11197889', '123741382#0'],
            '123741382#0': ['123741382#0', '123741382#1'],
            '123741382#1': ['123741382#1', '123741311'],
            '394443191': ['394443191'],
            '322962944': ['322962944'],
            '90077193#0': ['90077193#0', '90077193#1.0'],
            '11198599': ['11198599', '124952182.0'],
            '124952182.0': ['124952182.0', 'gneE0'],
            'gneE0': ['gneE0', '90077193#1.777'],
            '90077193#1.777': ['90077193#1.777', '90077193#1.812'],
            '32661309#0': ['32661309#0', '32661309#1.0'],
            '32661309#1.0': ['32661309#1.0', 'gneE1'],
            'gneE1': ['gneE1', '90077193#1.812'],
            '90077193#1.0': ['90077193#1.0', '90077193#1.777'],
            '90077193#1.812': ['90077193#1.812', 'gneE3'],
            'gneE3': ['gneE3', '340686911#0.54.0'],
            '340686911#0.54.0': ['340686911#0.54.0', '340686911#0.54.54.0'],
            '340686911#0.54.54.0':
                ['340686911#0.54.54.0', '340686911#0.54.54.127.0'],
            '340686911#0.54.54.127.0':
                ['340686911#0.54.54.127.0', '340686911#0.54.54.127.74'],
            '340686911#2.35': ['340686911#2.35', '340686911#3']
        }

        return rts
