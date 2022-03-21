# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Cartpole agent for impala algorithm."""

from xt.agent.ppo.cartpole_ppo import CartpolePpo
from xt.framework.register import Registers


@Registers.agent.register
class CartpoleImpala(CartpolePpo):
    """Cartpole Agent with Impala algorithm."""
    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        predict_val = self.alg.predict(next_raw_state)
        self.next_action = predict_val[0][0]
        self.next_value = predict_val[1][0]
        self.next_state = next_raw_state

        self.transition_data.update({
            "next_state": next_raw_state,
            "reward": reward,
            "next_value": self.next_value,
            "done": done,
            "info": info
        })

        return self.transition_data

    def add_to_trajectory(self, transition_data):
        for k, val in transition_data.items():
            if k is "next_state":
                self.trajectory.update({"last_state": val})
            else:
                if k not in self.trajectory.keys():
                    self.trajectory.update({k: [val]})
                else:
                    self.trajectory[k].append(val)

    def data_proc(self):
        pass

    def sync_model(self):
        model_name = "none"
        try:
            while True:
                model_name = self.recv_explorer.recv(block=False)
        except:
            pass
        return model_name
