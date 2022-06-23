# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Utility methods for playing an actual game as a human against a model.
"""
from npu_bridge.npu_init import *

import sys
from logging import getLogger

from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config, PlayWithHumanConfig
from chess_zero.env.chess_env import ChessEnv

logger = getLogger(__name__)

# noinspection SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection
def start(config: Config):

    PlayWithHumanConfig().update_play_config(config.play)

    me_player = None
    env = ChessEnv().reset()

    while True:
        line = input()
        words = line.rstrip().split(" ",1)
        if words[0] == "uci":
            print("id name ChessZero")
            print("id author ChessZero")
            print("uciok")
        elif words[0] == "isready":
            if not me_player:
                me_player = get_player(config)
            print("readyok")
        elif words[0] == "ucinewgame":
            env.reset()
        elif words[0] == "position":
            words = words[1].split(" ",1)
            if words[0] == "startpos":
                env.reset()
            else:
                if words[0] == "fen": # skip extraneous word
                    words = words[1].split(' ',1)
                fen = words[0]
                for _ in range(5):
                    words = words[1].split(' ',1)
                    fen += " " + words[0]
                env.update(fen)
            if len(words) > 1:
                words = words[1].split(" ",1)
                if words[0] == "moves":
                    for w in words[1].split(" "):
                        env.step(w, False)
        elif words[0] == "go":
            if not me_player:
                me_player = get_player(config)
            action = me_player.action(env, False)
            print(f"bestmove {action}")
        elif words[0] == "stop":
            pass
        elif words[0] == "quit":
            break


def get_player(config):
    from chess_zero.agent.model_chess import ChessModel
    from chess_zero.lib.model_helper import load_best_model_weight
    model = ChessModel(config)
    if not load_best_model_weight(model):
        raise RuntimeError("Best model not found!")
    return ChessPlayer(config, model.get_pipes(config.play.search_threads))


def info(depth, move, score):
    print(f"info score cp {int(score*100)} depth {depth} pv {move}")
    sys.stdout.flush()

