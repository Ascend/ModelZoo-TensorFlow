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
import cv2
import itertools
import math
import multiprocessing
import numpy as np
import numpy.linalg as npla
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import models
import samplelib
from DFLIMG import DFLIMG
from core import imagelib
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand, MPFunc
from core.leras import nn
from eval import FrameInfo, InteractiveEvalSubprocessor, MergerConfig
from facelib import FaceType, LandmarksProcessor


# from tensorflow.python.keras import backend as K
# from npu_bridge.npu_init import *


def trainerThread(s2c, c2s, e,
                  model_class_name=None,
                  saved_models_path=None,
                  training_data_src_path=None,
                  training_data_dst_path=None,
                  pretraining_data_path=None,
                  pretrained_model_path=None,

                  input_path=None,
                  output_path=None,
                  output_mask_path=None,

                  no_preview=False,
                  force_model_name=None,
                  force_gpu_idxs=None,
                  cpu_only=None,
                  silent_start=False,
                  execute_programs=None,
                  debug=False,
                  target_iters=0,
                  eval_iters=1000,
                  stop_SSIM=0.66,
                  **kwargs):
    start_time = time.time()

    if not training_data_src_path.exists():
        training_data_src_path.mkdir(exist_ok=True, parents=True)

    if not training_data_dst_path.exists():
        training_data_dst_path.mkdir(exist_ok=True, parents=True)

    if not saved_models_path.exists():
        saved_models_path.mkdir(exist_ok=True, parents=True)

    model = models.import_model(model_class_name)(
        is_training=True,
        saved_models_path=saved_models_path,
        training_data_src_path=training_data_src_path,
        training_data_dst_path=training_data_dst_path,
        pretraining_data_path=pretraining_data_path,
        pretrained_model_path=pretrained_model_path,
        no_preview=no_preview,
        force_model_name=force_model_name,
        force_gpu_idxs=force_gpu_idxs,
        cpu_only=cpu_only,
        silent_start=silent_start,
        debug=debug, )

    is_reached_goal = model.is_reached_iter_goal()  # True or False

    shared_state = {'after_save': False}
    loss_string = ""
    save_iter = model.get_iter()

    def model_save():
        if not debug and not is_reached_goal:
            io.log_info("Saving....", end='\r')
            model.save()
            shared_state['after_save'] = True

    def model_backup():
        if not debug and not is_reached_goal:
            model.create_backup()

    def send_preview():
        if not debug:
            previews = model.get_previews()
            c2s.put({'op': 'show', 'previews': previews, 'iter': model.get_iter(),
                     'loss_history': model.get_loss_history().copy()})
        else:
            previews = [('debug, press update for new', model.debug_one_iter())]
            c2s.put({'op': 'show', 'previews': previews})
        e.set()
        # Set the GUI Thread as Ready

    if model.get_target_iter() != 0:
        if is_reached_goal:
            io.log_info('Model already trained to target iteration. You can use preview.')
        else:
            io.log_info('Starting. Target iteration: %d.' % (model.get_target_iter()))
    else:
        io.log_info('Starting.')

    execute_programs = [[x[0], x[1], time.time()] for x in execute_programs]

    ##################
    # for evaluation #
    ##################
    aligned_path = training_data_dst_path
    if not input_path.exists():
        io.log_err('Input directory not found. Please ensure it exists.')
        return

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if not output_mask_path.exists():
        output_mask_path.mkdir(parents=True, exist_ok=True)

    predictor_func, predictor_input_shape, cfg = model.get_EvalConfig()
    predictor_func = MPFunc(predictor_func)  # Preparing MP functions

    run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0

    is_interactive = False
    if not is_interactive:
        cfg.ask_settings()

    s = "Number of workers for evaluation is set to " + str(min(16, multiprocessing.cpu_count())) + " as default. \n"
    io.log_info(s)
    subprocess_count = min(16, multiprocessing.cpu_count())
    input_path_image_paths = pathex.get_image_paths(input_path)
    frames = []

    if cfg.type == MergerConfig.TYPE_MASKED:
        if not aligned_path.exists():
            io.log_err('Aligned directory not found. Please ensure it exists.')
            return

        packed_samples = None
        try:
            packed_samples = samplelib.PackedFaceset.load(aligned_path)
        except:
            io.log_err(
                f"Error occured while loading samplelib.PackedFaceset.load {str(aligned_path)}, {traceback.format_exc()}")

        if packed_samples is not None:
            io.log_info("Using packed faceset.")

            def generator():
                for sample in io.progress_bar_generator(packed_samples, "Collecting alignments"):
                    filepath = Path(sample.filename)
                    yield filepath, DFLIMG.load(filepath, loader_func=lambda x: sample.read_raw_file())
        else:
            def generator():
                for filepath in io.progress_bar_generator(pathex.get_image_paths(aligned_path),
                                                          "Collecting alignments"):
                    filepath = Path(filepath)
                    yield filepath, DFLIMG.load(filepath)

        alignments = {}
        multiple_faces_detected = False

        for filepath, dflimg in generator():
            if dflimg is None or not dflimg.has_data():
                io.log_err(f"{filepath.name} is not a dfl image file")
                continue

            source_filename = dflimg.get_source_filename()
            if source_filename is None:
                continue

            source_filepath = Path(source_filename)
            source_filename_stem = source_filepath.stem

            if source_filename_stem not in alignments.keys():
                alignments[source_filename_stem] = []

            alignments_ar = alignments[source_filename_stem]
            alignments_ar.append((dflimg.get_source_landmarks(), filepath, source_filepath))

            if len(alignments_ar) > 1:
                multiple_faces_detected = True

        if multiple_faces_detected:
            io.log_info("")
            io.log_info("Warning: multiple faces detected. Only one alignment file should refer one source file.")
            io.log_info("")

        for a_key in list(alignments.keys()):
            a_ar = alignments[a_key]
            if len(a_ar) > 1:
                for _, filepath, source_filepath in a_ar:
                    io.log_info(f"alignment {filepath.name} refers to {source_filepath.name} ")
                io.log_info("")

            alignments[a_key] = [a[0] for a in a_ar]

        if multiple_faces_detected:
            io.log_info("It is strongly recommended to process the faces separatelly.")
            io.log_info("Use 'recover original filename' to determine the exact duplicates.")
            io.log_info("")

        total_frame = len(input_path_image_paths)
        eval_iterval = total_frame // 100
        for i in range(total_frame):
            if i % eval_iterval == 0:
                frames.append(
                    InteractiveEvalSubprocessor.Frame(frame_info=FrameInfo(filepath=Path(input_path_image_paths[i]),
                                                                           landmarks_list=alignments.get(
                                                                               Path(input_path_image_paths[i]).stem,
                                                                               None)))
                )
        if multiple_faces_detected:
            io.log_info("Warning: multiple faces detected. Motion blur will not be used.")
            io.log_info("")
        else:
            s = 256
            local_pts = [(s // 2 - 1, s // 2 - 1), (s // 2 - 1, 0)]  # center+up
            frames_len = len(frames)
            for i in io.progress_bar_generator(range(len(frames)), "Computing motion vectors"):
                fi_prev = frames[max(0, i - 1)].frame_info
                fi = frames[i].frame_info
                fi_next = frames[min(i + 1, frames_len - 1)].frame_info
                if len(fi_prev.landmarks_list) == 0 or \
                        len(fi.landmarks_list) == 0 or \
                        len(fi_next.landmarks_list) == 0:
                    continue

                mat_prev = LandmarksProcessor.get_transform_mat(fi_prev.landmarks_list[0], s,
                                                                face_type=FaceType.FULL)
                mat = LandmarksProcessor.get_transform_mat(fi.landmarks_list[0], s, face_type=FaceType.FULL)
                mat_next = LandmarksProcessor.get_transform_mat(fi_next.landmarks_list[0], s,
                                                                face_type=FaceType.FULL)

                pts_prev = LandmarksProcessor.transform_points(local_pts, mat_prev, True)
                pts = LandmarksProcessor.transform_points(local_pts, mat, True)
                pts_next = LandmarksProcessor.transform_points(local_pts, mat_next, True)

                motion_vector = pts_next[0] - pts_prev[0]
                fi.motion_power = npla.norm(motion_vector)

                motion_vector = motion_vector / fi.motion_power if fi.motion_power != 0 else np.array([0, 0],
                                                                                                      dtype=np.float32)

                fi.motion_deg = -math.atan2(motion_vector[1], motion_vector[0]) * 180 / math.pi

    while True:
        try:
            for i in itertools.count(0, 1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if prog_time > 0 and (cur_time - start_time) >= prog_time:
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time) >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("Unable to execute program: %s" % (prog))

                    if not is_reached_goal:
                        if model.get_iter() == 0:
                            io.log_info("")
                            io.log_info("Trying to do the first iteration. \
                                        If an error occurs, reduce the model parameters.")
                            io.log_info("")

                            if sys.platform[0:3] == 'win':
                                io.log_info("!!!")
                                io.log_info("Windows 10 users IMPORTANT notice. "
                                            "You should set this setting in order to work correctly.")
                                io.log_info("https://i.imgur.com/B7cmDCB.jpg")
                                io.log_info("!!!")

                        iter, iter_time = model.train_one_iter()
                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")
                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s]".format(time_str, iter, '{:0.4f}'.format(iter_time))
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms]".format(time_str, iter, int(iter_time * 1000))

                        if shared_state['after_save']:
                            shared_state['after_save'] = False

                            mean_loss = np.mean(loss_history[save_iter:iter], axis=0)

                            for loss_value in mean_loss:
                                loss_string += "[%.4f]" % (loss_value)

                            io.log_info(loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.4f]" % (loss_value)

                            if io.is_colab():
                                io.log_info('\r' + loss_string, end='')
                            else:
                                io.log_info(loss_string, end='\r')

                    if target_iters == 0:
                        if (model.get_iter() - save_iter) % eval_iters == 0:
                            if len(frames) == 0:
                                io.log_info("\nNo frames for evaluation.")
                                i = -1
                                c2s.put({'op': 'close'})
                            else:
                                SSIM = InteractiveEvalSubprocessor(is_interactive=is_interactive,
                                                                   merger_session_filepath=model.get_strpath_storage_for_file(
                                                                       'merger_session.dat'),
                                                                   predictor_func=predictor_func,
                                                                   predictor_input_shape=predictor_input_shape,
                                                                   merger_config=cfg,
                                                                   frames=frames,
                                                                   frames_root_path=input_path,
                                                                   output_path=output_path,
                                                                   output_mask_path=output_mask_path,
                                                                   model_iter=model.get_iter(),
                                                                   subprocess_count=subprocess_count,
                                                                   ).run()
                                s = '\nNow iteration is ' + str(model.get_iter()) + '\n'
                                if np.mean(SSIM) >= stop_SSIM:
                                    s += f'SSIM: mean [{np.mean(SSIM)}], std [{np.std(SSIM, ddof=1)}], Done\n'
                                    interval = cur_time - start_time
                                    minute, sec = divmod(interval, 60)
                                    hour, minute = divmod(minute, 60)
                                    s += f"Time consumption: [{int(hour)} hours {int(minute)} minutes {int(sec)} seconds]"
                                    io.log_info(s)
                                    model_save()
                                    send_preview()
                                    is_reached_goal = True
                                    i = -1
                                    c2s.put({'op': 'close', 'iteration': model.get_iter()})
                                else:
                                    s += f'SSIM: mean [{np.mean(SSIM)}], std [{np.std(SSIM, ddof=1)}], Continue\n'
                                    io.log_info(s)

                    elif model.get_iter() >= target_iters:
                        if len(frames) == 0:
                            io.log_info("\nNo frames for evaluation.")
                            i = -1
                            c2s.put({'op': 'close'})
                        else:
                            SSIM = InteractiveEvalSubprocessor(is_interactive=is_interactive,
                                                               merger_session_filepath=model.get_strpath_storage_for_file(
                                                                   'merger_session.dat'),
                                                               predictor_func=predictor_func,
                                                               predictor_input_shape=predictor_input_shape,
                                                               merger_config=cfg,
                                                               frames=frames,
                                                               frames_root_path=input_path,
                                                               output_path=output_path,
                                                               output_mask_path=output_mask_path,
                                                               model_iter=model.get_iter(),
                                                               subprocess_count=subprocess_count,
                                                               ).run()
                            s = f'\nTarget iteration: [{target_iters}] achieved\n'
                            s += f'SSIM: mean [{np.mean(SSIM)}], std [{np.std(SSIM, ddof=1)}], Done\n'
                            interval = cur_time - start_time
                            minute, sec = divmod(interval, 60)
                            hour, minute = divmod(minute, 60)
                            s += f"Time consumption: [{int(hour)} hours {int(minute)} minutes {int(sec)} seconds]"
                            io.log_info(s)
                            model_save()
                            send_preview()
                            is_reached_goal = True
                            i = -1
                            c2s.put({'op': 'close', 'iteration': model.get_iter()})

                        # if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                        #     io.log_info('Reached target iteration.')
                        #     model_save()
                        #     is_reached_goal = True
                        #     io.log_info('You can use preview now.')
                        #     c2s.put({'op': 'close'})

                # need_save = False
                # while time.time() - last_save_time >= save_interval_min * 60:
                #     last_save_time += save_interval_min * 60
                #     need_save = True
                #
                # if not is_reached_goal and need_save:
                #     model_save()
                #     send_preview()
                #
                # if i == 0:
                #     if is_reached_goal:
                #         model.pass_one_iter()
                #     send_preview()

                # if not is_reached_goal:
                #     send_preview()

                if debug:
                    time.sleep(0.005)

                while not s2c.empty():
                    input = s2c.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'backup':
                        model_backup()
                    elif op == 'preview':
                        if is_reached_goal:
                            model.pass_one_iter()
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break

                if i == -1:
                    break
            #
            # if not output_path.exists():
            #     shutil.rmtree(output_path)
            # if not output_mask_path.exists():
            #     shutil.rmtree(output_mask_path)
            model.finalize()

        except Exception as e:
            print('Error: %s' % (str(e)))
            traceback.print_exc()
        break

    c2s.put({'op': 'close'})


def main(**kwargs):
    io.log_info("Running trainer.\r\n")

    no_preview = kwargs.get('no_preview', False)

    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e), kwargs=kwargs)
    thread.start()

    e.wait()
    # Wait for inital load to occur.

    if no_preview:
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op')
                if op == 'close':
                    break
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put({'op': 'close'})
    else:
        wnd_name = "Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0
        iter = 0
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iter = input['iter'] if 'iter' in input.keys() else 0
                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            (h, w, c) = preview_rgb.shape
                            max_h = max(max_h, h)
                            max_w = max(max_w, w)

                        max_size = 800
                        if max_h > max_size:
                            max_w = int(max_w / (max_h / max_size))
                            max_h = max_size

                        # make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append((preview_name, cv2.resize(preview_rgb, (max_w, max_h))))
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False

                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                (h, w, c) = selected_preview_rgb.shape

                # HEAD
                head_lines = [
                    '[s]:save [b]:backup [enter]:exit',
                    '[p]:update [space]:next preview [l]:change history range',
                    'Preview: "%s" [%d/%d]' % (selected_preview_name, selected_preview + 1, len(previews))
                ]
                head_line_height = 15
                head_height = len(head_lines) * head_line_height
                head = np.ones((head_height, w, c)) * 0.1

                for i in range(0, len(head_lines)):
                    t = i * head_line_height
                    b = (i + 1) * head_line_height
                    head[t:b, 0:w] += imagelib.get_text_image((head_line_height, w, c), head_lines[i], color=[0.8] * c)

                final = head

                if loss_history is not None:
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]

                    lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)
                    final = np.concatenate([final, lh_img], axis=0)

                final = np.concatenate([final, selected_preview_rgb], axis=0)
                final = np.clip(final, 0, 1)

                io.show_image(wnd_name, (final * 255).astype(np.uint8))
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (
                0, 0, False, False, False)

            if key == ord('\n') or key == ord('\r'):
                s2c.put({'op': 'close'})
            elif key == ord('s'):
                s2c.put({'op': 'save'})
            elif key == ord('b'):
                s2c.put({'op': 'backup'})
            elif key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put({'op': 'preview'})
            elif key == ord('l'):
                if show_last_history_iters_count == 0:
                    show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                    show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0
                update_preview = True
            elif key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True

            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put({'op': 'close'})
        io.destroy_all_windows()
