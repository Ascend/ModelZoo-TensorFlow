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
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from synthesizer.hparams import hparams_debug_string
from synthesizer.feeder import Feeder
#from synthesizer.feeder import Feeder
from synthesizer.models import create_model
from synthesizer.utils import ValueWindow, plot
from synthesizer.utils.recorder import VarRecorder
from synthesizer import infolog, audio
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import traceback
import time
import os

log = infolog.log


def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
    # Create tensorboard projector
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = checkpoint_path
    
    for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
        # Initialize config
        embedding = config.embeddings.add()
        # Specifiy the embedding variable and the metadata
        embedding.tensor_name = embedding_name
        embedding.metadata_path = path_to_meta
    
    # Project the embeddings to space dimensions for visualization
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)


def add_train_stats(model, hparams):
    with tf.variable_scope("stats") as scope:
        for i in range(hparams.tacotron_num_gpus):
            tf.summary.histogram("mel_outputs %d" % i, model.tower_mel_outputs[i])
            tf.summary.histogram("mel_targets %d" % i, model.tower_mel_targets[i])
        tf.summary.scalar("before_loss", model.before_loss)
        tf.summary.scalar("after_loss", model.after_loss)
        
        if hparams.predict_linear:
            tf.summary.scalar("linear_loss", model.linear_loss)
            for i in range(hparams.tacotron_num_gpus):
                tf.summary.histogram("mel_outputs %d" % i, model.tower_linear_outputs[i])
                tf.summary.histogram("mel_targets %d" % i, model.tower_linear_targets[i])
        
        tf.summary.scalar("regularization_loss", model.regularization_loss)
        #tf.summary.scalar("stop_token_loss", model.stop_token_loss)
        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("learning_rate", model.learning_rate)  # Control learning rate decay speed
        
        #TODO: Commented following
        # if hparams.tacotron_teacher_forcing_mode == "scheduled":
        #     tf.summary.scalar("teacher_forcing_ratio", model.ratio)  # Control teacher forcing 
        # ratio decay when mode = "scheduled"
        # gradient_norms = [tf.norm(grad) for grad in model.gradients]
        # tf.summary.histogram("gradient_norm", gradient_norms)
        # tf.summary.scalar("max_gradient_norm", tf.reduce_max(gradient_norms))  # visualize 
        # gradients (in case of explosion)
        return tf.summary.merge_all()


def add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss,
                   loss):
    values = [
        tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_before_loss",
                         simple_value=before_loss),
        tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_after_loss",
                         simple_value=after_loss),
        tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/stop_token_loss",
                         simple_value=stop_token_loss),
        tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_loss", simple_value=loss),
    ]
    if linear_loss is not None:
        values.append(tf.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_linear_loss",
                                       simple_value=linear_loss))
    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def model_train_mode(args, feeder, hparams, global_step):
    with tf.variable_scope("Tacotron_model", reuse=tf.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.inputs_test, feeder.input_lengths_test, feeder.speaker_embeddings_test, 
                         feeder.mel_targets_test, targets_lengths=feeder.targets_lengths, global_step=global_step,
                         is_training=True, split_infos=feeder.split_infos_test)
        print ("Model is initialized....")
        model.add_loss()
        print ("Loss is added.....")
        model.test_attention_optimizer(global_step)
        ##TODO:
        #model.add_optimizer(global_step)
        print ("Optimizer is added....")
        stats = add_train_stats(model, hparams)
        return model, stats


def model_test_mode(args, feeder, hparams, global_step):
    with tf.variable_scope("Tacotron_model", reuse=tf.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, 
                         feeder.eval_speaker_embeddings, feeder.eval_mel_targets, targets_lengths=feeder.eval_targets_lengths, 
                         global_step=global_step, is_training=False, is_evaluating=True,
                         split_infos=feeder.eval_split_infos)
        model.add_loss()
        return model


def train(log_dir, args, hparams):
    save_dir = os.path.join(log_dir, "taco_pretrained")
    plot_dir = os.path.join(log_dir, "plots")
    wav_dir = os.path.join(log_dir, "wavs")
    mel_dir = os.path.join(log_dir, "mel-spectrograms")
    eval_dir = os.path.join(log_dir, "eval-dir")
    eval_plot_dir = os.path.join(eval_dir, "plots")
    eval_wav_dir = os.path.join(eval_dir, "wavs")
    tensorboard_dir = os.path.join(log_dir, "tacotron_events")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    checkpoint_fpath = os.path.join(save_dir, "tacotron_model.ckpt")
    
    log("Checkpoint path: {}".format(checkpoint_fpath))
    log("Using model: Tacotron")
    log(hparams_debug_string())
    
    # Start by setting a seed for repeatability
    tf.set_random_seed(hparams.tacotron_random_seed)
    
    # Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope("datafeeder") as scope:
        feeder = Feeder(coord, hparams)
    
    # Set up model:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    model, stats = model_train_mode(args, feeder, hparams, global_step)
    #eval_model = model_test_mode(args, feeder, hparams, global_step)
    
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=2)
    
    log("Tacotron training set to a maximum of {} steps".format(args.tacotron_train_steps))
    
    # Memory allocation on the GPU as needed
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True

    #############################

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision") #allow_fp32_to_fp16

    #config = tf.compat.as_bytes("allow_mix_precision") npu_config_proto(config_proto=config_proto)
    #added for profiling
    if False:
        custom_op.parameter_map["use_off_line"].b = True
        #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["profiling_mode"].b = True
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/jiayansuo/Lip2wav_train/profiling","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"Tacotron_model/inference/encoder_convolutions/conv_layer_1_1_encoder_convolutions/conv3d/Conv3D","bp_point":"Tacotron_model/gradients/AddN_105"}') # Tacotron_model/inference/encoder_convolutions/conv_layer_1_encoder_convolutions/conv3d/Conv3D Tacotron_model/gradients/AddN_105  Tacotron_model/gradients/Tacotron_model/inference/encoder_convolutions/conv_layer_1_encoder_convolutions/conv3d/Conv3D_grad/tuple/control_dependency_1
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Remapping must be disabled explicitly.
        
    # Train
    with tf.Session(config = config) as sess:     
        #added for profiling
        #tf.io.write_graph(sess.graph_def, './', 'graph_profiling08-10.pbtxt')
    #with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
        try:
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            
            #print ("Restore val : ", args.restore)
            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    
                    if checkpoint_state and checkpoint_state.model_checkpoint_path:
                        # log("Loading checkpoint {}".format(checkpoint_state.model_checkpoint_path),
                        #     slack=True)
                        # saver.restore(sess, checkpoint_state.model_checkpoint_path)
                        log("Loading checkpoint {}".format(hparams.eval_ckpt),
                            slack=True)
                        saver.restore(sess, hparams.eval_ckpt)
                    
                    else:
                        log("No model to load at {}".format(save_dir), slack=True)
                        saver.save(sess, checkpoint_fpath, global_step=global_step)
                
                except tf.errors.OutOfRangeError as e:
                    log("Cannot restore checkpoint: {}".format(e), slack=True)
            else:
                log("Starting new training!", slack=True)
                saver.save(sess, checkpoint_fpath, global_step=global_step)
            
            # initializing feeder - enqueue in background thread
            feeder.start_threads(sess)
            print ("Feeder is initialized....")
            print ("Ready to train....")
            
            # Training loop
            time.sleep(10)
            while not coord.should_stop() and step < args.tacotron_train_steps:
                start_time = time.time()
                #time.sleep(10) # try this to see if thread queues feed_dict before running the snippet below
            
                # Or potentially manually load a feed_dict to the queue during first run
                feed_dict = None

                if not feeder.q.empty():
                    feed_dict = feeder.q.get()
                    #print("Queue not empty! Dequeue to fetch feed_dict and pass to sess.run")
                    # print ('Dequeue feed_dict output: feed dict : ', len(feed_dict))
                else:
                    #print("Queue empty, fetching from disk...")
                    feed_dict = feeder.get_feed_dict()
                    # print ('get_feed_dict output : feed dict : ', len(feed_dict))
                load_data_time = time.time()
                # ###############################################################
                # Verify feed_dict keys & values (are they real np values)
                # ###############################################################
                # feed_dict.keys() = 6 tf.placeholder defined in Feeder._placeholder
                # feed_dict.values() = batch 
                #print("Feed_dict")

                input_key_list = list(feed_dict.keys())
                input_val_list = list(feed_dict.values())
                
                assert global_step is not None, "global_step is None"
                assert model.loss is not None, "model.loss is None"
                assert model.optimize is not None, "model.optimize is None"

    #           step,_ = sess.run([global_step, ops], feed_dict=feed_dict)
                
                #step, opt = sess.run([global_step, model.optimize], feed_dict=feed_dict)
                step, loss, opt = sess.run([global_step, model.loss, model.optimize], feed_dict=feed_dict)
                train_step_time = time.time()
                time_window.append(train_step_time - start_time)
                loss_window.append(loss)
                message = "Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}, load_time={:.3f}, train_time={:.3f}]".format(
                    step, time_window.average, loss, loss_window.average, load_data_time- start_time, train_step_time - load_data_time)
                log(message, end="\r", slack=(step % args.checkpoint_interval == 0))
                print(message)

                #print ("args vals: ", step, args.summary_interval)
                
                if loss > 100 or np.isnan(loss):
                    log("Loss exploded to {:.5f} at step {}".format(loss, step))
                    raise Exception("Loss exploded")
                
                #TODO: Commented
                if step % args.summary_interval == 0:
                    log("\nWriting summary at step {}".format(step))
                    summary_writer.add_summary(sess.run(stats, feed_dict=feed_dict), step)
                
                if step % args.eval_interval == 0:
                    pass
                    # Run eval and save eval stats
       #              log("\nRunning evaluation at step {}".format(step))
                    
       #              eval_losses = []
       #              before_losses = []
       #              after_losses = []
       #              stop_token_losses = []
       #              linear_losses = []
       #              linear_loss = None
                    
       #              if hparams.predict_linear:
       #                  for i in tqdm(range(feeder.test_steps)):
       #                      eloss, before_loss, after_loss, stop_token_loss, linear_loss, mel_p, \
							# mel_t, t_len, align, lin_p, lin_t = sess.run(
       #                          [
       #                              eval_model.tower_loss[0], eval_model.tower_before_loss[0],
       #                              eval_model.tower_after_loss[0],
       #                              eval_model.tower_stop_token_loss[0],
       #                              eval_model.tower_linear_loss[0],
       #                              eval_model.tower_mel_outputs[0][0],
       #                              eval_model.tower_mel_targets[0][0],
       #                              eval_model.tower_targets_lengths[0][0],
       #                              eval_model.tower_alignments[0][0],
       #                              eval_model.tower_linear_outputs[0][0],
       #                              eval_model.tower_linear_targets[0][0],
       #                          ])
       #                      eval_losses.append(eloss)
       #                      before_losses.append(before_loss)
       #                      after_losses.append(after_loss)
       #                      stop_token_losses.append(stop_token_loss)
       #                      linear_losses.append(linear_loss)
       #                  linear_loss = sum(linear_losses) / len(linear_losses)
                        
       #                  wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
       #                  audio.save_wav(wav, os.path.join(eval_wav_dir,
       #                                                   "step-{}-eval-wave-from-linear.wav".format(
       #                                                       step)), sr=hparams.sample_rate)
                    
       #              else:
       #                  for i in tqdm(range(feeder.test_steps)):
       #                      eloss, before_loss, after_loss, stop_token_loss, mel_p, mel_t, t_len,\
							# align = sess.run(
       #                          [
       #                              eval_model.tower_loss[0], eval_model.tower_before_loss[0],
       #                              eval_model.tower_after_loss[0],
       #                              eval_model.tower_stop_token_loss[0],
       #                              eval_model.tower_mel_outputs[0][0],
       #                              eval_model.tower_mel_targets[0][0],
       #                              eval_model.tower_targets_lengths[0][0],
       #                              eval_model.tower_alignments[0][0]
       #                          ])
       #                      eval_losses.append(eloss)
       #                      before_losses.append(before_loss)
       #                      after_losses.append(after_loss)
       #                      stop_token_losses.append(stop_token_loss)
                    
       #              eval_loss = sum(eval_losses) / len(eval_losses)
       #              before_loss = sum(before_losses) / len(before_losses)
       #              after_loss = sum(after_losses) / len(after_losses)
       #              stop_token_loss = sum(stop_token_losses) / len(stop_token_losses)
                    
       #              log("Saving eval log to {}..".format(eval_dir))
       #              # Save some log to monitor model improvement on same unseen sequence
       #              wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
       #              audio.save_wav(wav, os.path.join(eval_wav_dir,
       #                                               "step-{}-eval-wave-from-mel.wav".format(step)),
       #                             sr=hparams.sample_rate)
                    
       #              plot.plot_alignment(align, os.path.join(eval_plot_dir,
       #                                                      "step-{}-eval-align.png".format(step)),
       #                                  title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
       #                                                                              time_string(),
       #                                                                              step,
       #                                                                              eval_loss),
       #                                  max_len=t_len // hparams.outputs_per_step)
       #              plot.plot_spectrogram(mel_p, os.path.join(eval_plot_dir,
       #                                                        "step-{"
							# 								  "}-eval-mel-spectrogram.png".format(
       #                                                            step)),
       #                                    title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
       #                                                                                time_string(),
       #                                                                                step,
       #                                                                                eval_loss),
       #                                    target_spectrogram=mel_t,
       #                                    max_len=t_len)
                    
       #              if hparams.predict_linear:
       #                  plot.plot_spectrogram(lin_p, os.path.join(eval_plot_dir,
       #                                                            "step-{}-eval-linear-spectrogram.png".format(
       #                                                                step)),
       #                                        title="{}, {}, step={}, loss={:.5f}".format(
       #                                            "Tacotron", time_string(), step, eval_loss),
       #                                        target_spectrogram=lin_t,
       #                                        max_len=t_len, auto_aspect=True)
                    
       #              log("Eval loss for global step {}: {:.3f}".format(step, eval_loss))
       #              log("Writing eval summary!")
       #              add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss,
       #                             stop_token_loss, eval_loss)
                
                if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or \
                        step == 300:
                    # Save model and current global step
                    saver.save(sess, checkpoint_fpath, global_step=global_step) #TODO:Commented
                    
                    log("\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..")

                    # Commented for mixed prec training              
                    # input_seq, mel_prediction, target, target_length = sess.run([ 
                    #     model.tower_inputs[0][0],
                    #     model.tower_mel_outputs[0][0],
                    #     #model.tower_alignments[0][0],
                    #     model.tower_mel_targets[0][0],
                    #     model.tower_targets_lengths[0][0],
                    # ], feed_dict = feed_dict) #alignment
                    
                    # #save predicted mel spectrogram to disk (debug)
                    # mel_filename = "mel-prediction-step-{}.npy".format(step)
                    # np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T,
                    #         allow_pickle=False)
                    
                    # # save griffin lim inverted wav for debug (mel -> wav)
                    # wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
                    # audio.save_wav(wav,
                    #                os.path.join(wav_dir, "step-{}-wave-from-mel.wav".format(step)),
                    #                sr=hparams.sample_rate)
                    
                    # # save alignment plot to disk (control purposes)
                    # # plot.plot_alignment(alignment,
                    # #                     os.path.join(plot_dir, "step-{}-align.png".format(step)),
                    # #                     title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                    # #                                                                 time_string(),
                    # #                                                                 step, loss),
                    # #                     max_len=target_length // hparams.outputs_per_step)
                    # # save real and predicted mel-spectrogram plot to disk (control purposes)
                    # plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir,
                    #                                                    "step-{}-mel-spectrogram.png".format(
                    #                                                        step)),
                    #                       title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                    #                                                                   time_string(),
                    #                                                                   step, loss),
                    #                       target_spectrogram=target,
                    #                       max_len=target_length)
                    # # log("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
                
                if step % args.embedding_interval == 0 or step == args.tacotron_train_steps or step == 1:
                    # Get current checkpoint state
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    
                    # Update Projector
                    #log("\nSaving Model Character Embeddings visualization..")
                    #add_embedding_stats(summary_writer, [model.embedding_table.name],
                    #                   [char_embedding_meta],
                    #                    checkpoint_state.model_checkpoint_path)
                    #log("Tacotron Character embeddings have been updated on tensorboard!")
            
            log("Tacotron training complete after {} global steps!".format(
                args.tacotron_train_steps), slack=True)
            return save_dir
        
        except Exception as e:
            log("Exiting due to exception: {}".format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def tacotron_train(args, log_dir, hparams):
    return train(log_dir, args, hparams)