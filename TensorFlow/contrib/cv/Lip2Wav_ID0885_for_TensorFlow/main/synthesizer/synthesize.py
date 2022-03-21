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


from npu_bridge.npu_init import *
from synthesizer.tacotron2 import Tacotron2
from synthesizer.hparams import hparams_debug_string
from synthesizer.infolog import log
import tensorflow as tf
from tqdm import tqdm
import time
import os


def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
    eval_dir = os.path.join(output_dir, "eval")
    log_dir = os.path.join(output_dir, "logs-eval")
    
    #Create output path if it doesn"t exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
    
    log(hparams_debug_string())
    synth = Tacotron2(checkpoint_path, hparams)
    
    #Set inputs batch wise
    sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i 
                 in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]
    
    log("Starting Synthesis")
    with open(os.path.join(eval_dir, "map.txt"), "w") as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            basenames = ["batch_{}_sentence_{}".format(i, j) for j in range(len(texts))]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)
            
            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write("|".join([str(x) for x in elems]) + "\n")
    log("synthesized mel spectrograms at {}".format(eval_dir))
    return eval_dir

def run_synthesis(in_dir, out_dir, model_dir, hparams):
    synth_dir = os.path.join(out_dir, "mels_gta")
    os.makedirs(synth_dir, exist_ok=True)
    metadata_filename = os.path.join(in_dir, "train.txt")
    print(hparams_debug_string())
    
    # Load the model in memory
    weights_dir = os.path.join(model_dir, "taco_pretrained")
    checkpoint_fpath = tf.train.get_checkpoint_state(weights_dir).model_checkpoint_path
    checkpoint_fpath = checkpoint_fpath.replace('/ssd_scratch/cvit/rudra/SV2TTS/', '')
    checkpoint_fpath = checkpoint_fpath.replace('logs-', '')
    synth = Tacotron2(checkpoint_fpath, hparams, gta=True)
    
    # Load the metadata
    with open(metadata_filename, encoding="utf-8") as f:
        metadata = [line.strip().split("|") for line in f][:149736]
        frame_shift_ms = hparams.hop_size / hparams.sample_rate
        hours = sum([int(x[4]) for x in metadata]) * frame_shift_ms / 3600
        print("Loaded metadata for {} examples ({:.2f} hours)".format(len(metadata), hours))
        
    #Set inputs batch wise
    metadata = [metadata[i: i + hparams.tacotron_synthesis_batch_size] for i in
                range(0, len(metadata), hparams.tacotron_synthesis_batch_size)]
    # TODO: come on big boy, fix this
    # Quick and dirty fix to make sure that all batches have the same size 
    metadata = metadata[:-1]
    
    print("Starting Synthesis")
    mel_dir = os.path.join(in_dir, "mels")
    embed_dir = os.path.join(in_dir, "embeds")
    meta_out_fpath = os.path.join(out_dir, "synthesized.txt")
    with open(meta_out_fpath, "w") as file:
        for i, meta in enumerate(tqdm(metadata)):
            texts = [m[5] for m in meta]
            mel_filenames = [os.path.join(mel_dir, m[1]) for m in meta]
            embed_filenames = [os.path.join(embed_dir, m[2]) for m in meta]
            basenames = [os.path.basename(m).replace(".npy", "").replace("mel-", "") 
                         for m in mel_filenames]
            synth.synthesize(texts, basenames, synth_dir, None, mel_filenames, embed_filenames)
            
            for elems in meta:
                file.write("|".join([str(x) for x in elems]) + "\n")
                
    print("Synthesized mel spectrograms at {}".format(synth_dir))
    return meta_out_fpath


