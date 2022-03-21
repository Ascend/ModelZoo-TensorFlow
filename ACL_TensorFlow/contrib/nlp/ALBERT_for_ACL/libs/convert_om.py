# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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

import datetime
import subprocess

from absl import flags

FLAGS = flags.FLAGS


def atc(mode=None,
        model=None,
        weight=None,
        om=None,
        framework=None,
        input_format=None,
        input_shape=None,
        dynamic_batch_size=None,
        dynamic_image_size=None,
        dynamic_dims=None,
        single_op=None,
        output=None,
        output_type=None,
        check_report=None,
        json_file=None,
        soc_version=None,
        core_type=None,
        aicore_num=None,
        out_nodes=None,
        input_fp16_nodes=None,
        insert_op_conf=None,
        op_name_map=None,
        is_input_adjust_hw_layout=None,
        is_output_adjust_hw_layout=None,
        disable_reuse_memory=None,
        fusion_switch_file=None,
        enable_scope_fusion_passes=None,
        enable_single_stream=None,
        enable_small_channel=None,
        enable_compress_weight=None,
        compress_weight_conf=None,
        buffer_optimize=None,
        precision_mode=None,
        auto_tune_mode=None,
        op_select_implmode=None,
        optypelist_for_implmode=None,
        op_debug_level=None,
        keep_dtype=None,
        save_original_model=None,
        log=None,
        dump_mode=None,
        debug_dir=None,
        op_compiler_cache_dir=None,
        op_compiler_cache_mode=None):
    # support atc interface, all parameters details below
    input_dict = {
        # ===== Basic Functionality =====
        # General
        "--mode=": mode,
        # Input
        "--model=": model, "--weight=": weight, "--om=": om, "--framework=": framework, "--input_format=": input_format,
        "--input_shape=": input_shape, "--dynamic_batch_size=": dynamic_batch_size,
        "--dynamic_image_size=": dynamic_image_size, "--dynamic_dims=": dynamic_dims, "--singleop=": single_op,
        # Output
        "--output=": output, "--output_type=": output_type, "--check_report=": check_report, "--json=": json_file,
        # Target
        "--soc_version=": soc_version, "--core_type=": core_type, "--aicore_num=": aicore_num,
        # ===== Advanced Functionality =====
        # Feature
        "--out_nodes=": out_nodes, "--input_fp16_nodes=": input_fp16_nodes, "--insert_op_conf=": insert_op_conf,
        "--op_name_map=": op_name_map, "--is_input_adjust_hw_layout=": is_input_adjust_hw_layout,
        "--is_output_adjust_hw_layout=": is_output_adjust_hw_layout,
        # Model Tuning
        "--disable_reuse_memory=": disable_reuse_memory, "--fusion_switch_file=": fusion_switch_file,
        "--enable_scope_fusion_passes=": enable_scope_fusion_passes, "--enable_single_stream=": enable_single_stream,
        "--enable_small_channel=": enable_small_channel, "--enable_compress_weight=": enable_compress_weight,
        "--compress_weight_conf=": compress_weight_conf, "--buffer_optimize=": buffer_optimize,
        # Operator Tuning
        "--precision_mode=": precision_mode, "--auto_tune_mode=": auto_tune_mode,
        "--op_select_implmode=": op_select_implmode, "--optypelist_for_implmode=": optypelist_for_implmode,
        "--op_debug_level=": op_debug_level, "--keep_dtype=": keep_dtype,
        # Debug
        "--save_original_model=": save_original_model, "--log=": log, "--dump_mode=": dump_mode,
        "--debug_dir=": debug_dir, "--op_compiler_cache_dir=": op_compiler_cache_dir,
        "--op_compiler_cache_mode=": op_compiler_cache_mode
    }
    output_list = []
    for input_key in input_dict.keys():
        if input_dict[input_key] is not None:
            output_list.append(input_key + str(input_dict[input_key]))
    cmd = "atc %s" % " ".join(output_list)
    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                    "I", "ATC cmd: %s" % cmd))
    subprocess.call(cmd, shell=True, stderr=subprocess.PIPE, timeout=300)


def convert_om(infer_param):
    atc_input_dict = {
        "mode": None,
        "model": None,
        "weight": None,
        "om": None,
        "framework": None,
        "input_format": None,
        "input_shape": None,
        "dynamic_batch_size": None,
        "dynamic_image_size": None,
        "dynamic_dims": None,
        "single_op": None,
        "output": None,
        "output_type": None,
        "check_report": None,
        "json": None,
        "soc_version": None,
        "core_type": None,
        "aicore_num": None,
        "out_nodes": None,
        "input_fp16_nodes": None,
        "insert_op_conf": None,
        "op_name_map": None,
        "is_input_adjust_hw_layout": None,
        "is_output_adjust_hw_layout": None,
        "disable_reuse_memory": None,
        "fusion_switch_file": None,
        "enable_scope_fusion_passes": None,
        "enable_single_stream": None,
        "enable_small_channel": None,
        "enable_compress_weight": None,
        "compress_weight_conf": None,
        "buffer_optimize": None,
        "precision_mode": None,
        "auto_tune_mode": None,
        "op_select_implmode": None,
        "optypelist_for_implmode": None,
        "op_debug_level": None,
        "keep_dtype": None,
        "save_original_model": None,
        "log": None,
        "dump_mode": None,
        "debug_dir": None,
        "op_compiler_cache_dir": None,
        "op_compiler_cache_mode": None
    }
    for key in infer_param.keys():
        atc_input_dict.update({key: infer_param[key]})

    atc(mode=atc_input_dict["mode"],
        model=atc_input_dict["model"],
        weight=atc_input_dict["weight"],
        om=atc_input_dict["om"],
        framework=atc_input_dict["framework"],
        input_format=atc_input_dict["input_format"],
        input_shape=atc_input_dict["input_shape"],
        dynamic_batch_size=atc_input_dict["dynamic_batch_size"],
        dynamic_image_size=atc_input_dict["dynamic_image_size"],
        dynamic_dims=atc_input_dict["dynamic_dims"],
        single_op=atc_input_dict["single_op"],
        output=atc_input_dict["output"],
        output_type=atc_input_dict["output_type"],
        check_report=atc_input_dict["check_report"],
        json_file=atc_input_dict["json"],
        soc_version=atc_input_dict["soc_version"],
        core_type=atc_input_dict["core_type"],
        aicore_num=atc_input_dict["aicore_num"],
        out_nodes=atc_input_dict["out_nodes"],
        input_fp16_nodes=atc_input_dict["input_fp16_nodes"],
        insert_op_conf=atc_input_dict["insert_op_conf"],
        op_name_map=atc_input_dict["op_name_map"],
        is_input_adjust_hw_layout=atc_input_dict["is_input_adjust_hw_layout"],
        is_output_adjust_hw_layout=atc_input_dict["is_output_adjust_hw_layout"],
        disable_reuse_memory=atc_input_dict["disable_reuse_memory"],
        fusion_switch_file=atc_input_dict["fusion_switch_file"],
        enable_scope_fusion_passes=atc_input_dict["enable_scope_fusion_passes"],
        enable_single_stream=atc_input_dict["enable_single_stream"],
        enable_small_channel=atc_input_dict["enable_small_channel"],
        enable_compress_weight=atc_input_dict["enable_compress_weight"],
        compress_weight_conf=atc_input_dict["compress_weight_conf"],
        buffer_optimize=atc_input_dict["buffer_optimize"],
        precision_mode=atc_input_dict["precision_mode"],
        auto_tune_mode=atc_input_dict["auto_tune_mode"],
        op_select_implmode=atc_input_dict["op_select_implmode"],
        optypelist_for_implmode=atc_input_dict["optypelist_for_implmode"],
        op_debug_level=atc_input_dict["op_debug_level"],
        keep_dtype=atc_input_dict["keep_dtype"],
        save_original_model=atc_input_dict["save_original_model"],
        log=atc_input_dict["log"],
        dump_mode=atc_input_dict["dump_mode"],
        debug_dir=atc_input_dict["debug_dir"],
        op_compiler_cache_dir=atc_input_dict["op_compiler_cache_dir"],
        op_compiler_cache_mode=atc_input_dict["op_compiler_cache_mode"])
