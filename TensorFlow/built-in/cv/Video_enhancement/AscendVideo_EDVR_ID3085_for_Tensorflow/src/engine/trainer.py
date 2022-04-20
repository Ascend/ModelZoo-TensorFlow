# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import time

import numpy as np
import tensorflow as tf
from src.losses.modules.perceptual import load_perceptual_module
from src.runner.common import name_space
from src.runner.distributed_variables_broadcast import \
    broadcast_global_variables, allreduce_avg
from src.runner.helper import build_adversarial_train_helper
from src.runner.saver import strict_loading, loose_loading
from src.runner.sess_config import get_sess_config
from src.runner.solver import build_solver
from src.utils.exceptions import *
from src.utils.logger import logger
from src.utils.moving_avg import MovingAvg
from src.utils.world import world


class _Trainer:
    """Base trainer class.
    This class is for tensorflow for now.

    Args:
        dataloader: list[tensor] generated from tf dataloader. See `src.dataloaders.dataloder`
            for more information.
        network: network instance whose class derives from Base network class.
        cfg: yacs node, global configuration.
        _world: World instance, option reserved for extension. By default, the trainer uses a 
            preset global `world` instance.
    """
    def __init__(self, dataloader, network, cfg, _world=None):
        self.device = cfg.env.device
        self.is_distributed = cfg.env.rank_size > 1
        self.cfg = cfg

        self.dataloader = dataloader
        self.network = network
        self.g_train_op = None
        self.d_train_op = None
        self.g_solver = None
        self.d_solver = None

        self.step_time = MovingAvg(smooth=0.9)
        self.step_loss = MovingAvg(smooth=0.99)

        self.world = _world or world
        if not self.world.is_initialized:
            raise WorldUninitializedError('World not initialized.')

        # Call network.build_graph to construct the basic graph.
        # Including dataloader, forward graph, and loss
        self.network.build_graph(dataloader=self.dataloader)

        # Helper is to coordinate the adversarial training, i.e.,
        # whether to update the generator or the discriminator according to
        # certain strategy.
        self.helper = build_adversarial_train_helper(cfg)

        # Build the optimizers
        self.build()

    def build(self):
        """
        Top building function to prepare optimizers.
        """
        self.build_g_optimizer()

        # Prepare discriminator optimizer if required
        if self.cfg.loss.adversarial.loss_weight > 0.:
            self.build_d_optimizer()

        # Use GLOBAL_VARIABLES to get both the weights and buffers.
        # Do not use tf.GraphKeys.TRAINABLE_VARIABLES here, which will miss the 
        # bn buffers.
        generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope=self.cfg.model.scope)
        self.saver = tf.train.Saver(var_list=generator_vars,
                                    max_to_keep=10,
                                    keep_checkpoint_every_n_hours=1)

    def build_g_optimizer(self):
        """
        Build generator optimizer.
        """
        # Build generator solver
        self.g_solver = build_solver(self.cfg.train.generator.lr_schedule,
                                     self.cfg.train.optimizer,
                                     self.cfg.session.mix_precision,
                                     self.cfg.train.loss_scale,
                                     self.device,
                                     self.is_distributed)
        
        # All generator losses are collected in name_space.GeneratorLoss scope.
        # Add them to get the final generator loss.
        losses_dict = name_space.get_collection(name_space.GeneratorLoss)
        losses = tf.add_n(list(losses_dict.values()))

        name_space.add_to_collection(name_space.GeneratorLoss, 'loss_total', losses)
        # TODO: encapsulate the learning rate
        name_space.add_to_collection(name_space.GeneratorRunOp, 'g_lr', self.g_solver.lr)

        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           scope=self.cfg.model.scope)

        g_train_op = self.g_solver.opt.minimize(losses, var_list=generator_vars)

        # bn buffer update after the optimization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 
                                       scope=self.cfg.model.scope)
        if list(update_ops):
            with tf.control_dependencies([g_train_op]):
                g_train_op = tf.group(*update_ops)

        self.g_train_op = g_train_op

        # Add to name_space for later query
        name_space.add_to_collection(name_space.GeneratorRunOp, 'g_train', self.g_train_op)

    def build_d_optimizer(self):
        """
        Build discriminator optimizer.
        """
        self.d_solver = build_solver(self.cfg.train.discriminator.lr_schedule,
                                     self.cfg.train.optimizer,
                                     self.cfg.session.mix_precision,
                                     self.cfg.train.loss_scale,
                                     self.device,
                                     self.is_distributed)
        # All discriminator losses are collected in name_space.DiscriminatorLoss scope.
        # Add them to get the final discriminator loss.
        losses_dict = name_space.get_collection(name_space.DiscriminatorLoss)

        self.d_loss = tf.add_n(list(losses_dict.values()))
        name_space.add_to_collection(name_space.DiscriminatorRunOp, 'd_lr', self.d_solver.lr)

        discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=name_space.DiscriminatorVarScope)
        d_train_op = self.d_solver.opt.minimize(self.d_loss, var_list=discriminator_vars)
        
        # If parameter clip is applied, do it after optimization.
        if self.cfg.loss.adversarial.parameter_clip:
            amin, amax = self.cfg.loss.adversarial.parameter_clip_range
            with tf.control_dependencies([d_train_op]):
                d_train_op = tf.group([var.assign(tf.clip_by_value(var, amin, amax))
                                       for var in discriminator_vars])

        # bn buffer update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name_space.DiscriminatorVarScope)
        if list(update_ops):
            with tf.control_dependencies([d_train_op]):
                d_train_op = tf.group(*update_ops)

        self.d_train_op = d_train_op
        name_space.add_to_collection(name_space.DiscriminatorRunOp, 'd_train', self.d_train_op)

    def save(self, *args, **kwargs):
        """
        Save ckpt during training every certain steps.
        """
        raise NotImplementedError

    def print(self, *args, **kwargs):
        """
        Print function to dump train information.
        """
        raise NotImplementedError

    def restore(self):
        """
        Restore from a ckpt for continual training.
        """
        raise NotImplementedError

    def load_pretrained(self, *args, **kwargs):
        """
        Load pretrained sub-networks for overall training and fine-tune.
        """
        raise NotImplementedError

    def run(self):
        """
        Execute function to run the train steps.
        """
        raise NotImplementedError
    

class SessionTrainer(_Trainer):
    """ 
    Tensorflow trainer using tf.Session.
    """
    def __init__(self, dataloader, network, cfg):
        super().__init__(dataloader, network, cfg)
        sess_cfg = get_sess_config(cfg.env.device,
                                   cfg.session.xla,
                                   cfg.session.mix_precision,
                                   cfg.env.rank_size>1)
        self.session = tf.Session(config=sess_cfg)

        # TODO: refactor summary
        if cfg.train.use_tensorboard:  # for visualization when training drm
            self.writer = tf.summary.FileWriter(
                os.path.join(self.cfg.train.output_dir, 'summary'),
                self.session.graph
                )
        else:
            self.writer = None

    def save(self, step):
        """ 
        Save checkpoint on the step

        Args:
            step: int
        """
        if not os.path.exists(self.cfg.train.output_dir):
            os.makedirs(self.cfg.train.output_dir)

        self.saver.save(self.session, 
                        os.path.join(self.cfg.train.output_dir, 
                                     self.cfg.model.name),
                        global_step=step)

    def print(self, step, ops_result, loss_ops_result):
        """ 
        Print train step information on the screen

        Args:
            step: int, current train step.
            ops_result: dict, data obtained by session.run.
            loss_ops_result: dict, loss information
        """
        loss_str = [f'{k}: {f"{v:3.3f}":>7}' for k, v in loss_ops_result.items()]

        fps = (self.cfg.data.train.batch_size 
                / (self.step_time.cur_val + 1e-6)
                * self.cfg.env.rank_size)
        eta = (self.g_solver.total_step - step) * self.step_time.avg
        eta = str(datetime.timedelta(seconds=int(eta)))

        solver_info = [f'Step: [{step:>7d} / {self.g_solver.total_step}]']
        # If adversarial, print whether generator or discriminator is updated
        if self.cfg.loss.adversarial.loss_weight > 0.:
            adv_info = self.helper.info
            g_update = adv_info['g_update']
            d_update = adv_info['d_update']

            solver_info.append(f'g update: {f"{g_update}":>5}')
            g_lr = ops_result['g_lr']
            solver_info.append(f'g lr: {f"{g_lr:.7f}":>6}')

            solver_info.append(f'd update: {f"{d_update}":>5}')
            d_lr = ops_result['d_lr']
            solver_info.append(f'd lr: {f"{d_lr:.7f}":>6}')
        else:
            g_lr = ops_result['g_lr']
            solver_info.append(f'g lr: {f"{g_lr:.7f}":>6}')

        misc_info = [f'smooth_total: {f"{self.step_loss.smooth_avg:3.3f}":>7}',
                     f'step time: {f"{self.step_time.cur_val*1000:5.1f}":>7} ms',
                     f'fps: {f"{fps:3.2f}":>6}',
                     f'eta: {eta:>8}',
                     f'on device: {self.world.device_id:1d}']

        print_info = ', '.join([*solver_info, *loss_str, *misc_info])
        logger.info(print_info)

    def load_pretrained(self, scope):
        """ 
        Load part of the graph.

        This function is typically used in fine-tune, multi-stage training 
        scenarios.

        Args:
            scope: str, top scope name for pretrained sub-graph.
        """
        if self.cfg.checkpoint == '' and (
            len(self.cfg.train.pretrained_scope_list) > 0
            ):
            assert len(self.cfg.train.pretrained_scope_list) == \
                   len(self.cfg.model.pretrained_scope_ckpt)
            for scope, ckpt in zip(self.cfg.train.pretrained_scope_list, 
                                   self.cfg.model.pretrained_scope_ckpt):
                loose_loading(self.session, scope, 
                              self.cfg.train.output_dir, ckpt)
        else:
            loose_loading(self.session, self.cfg.model.scope, 
                          self.cfg.train.output_dir, self.cfg.checkpoint)

    def restore(self):
        """ 
        Restore ckpt.

        This function is for continue training scenario. Thus every thing in 
        the generator will be loaded.

        Returns:
            int, recover iteration to continue training.
        """
        return strict_loading(self.session, 
                              self.cfg.model.scope,
                              self.cfg.train.output_dir, 
                              self.cfg.checkpoint)

    def run(self):
        """ 
        Core function for the trainer to execute.
        """
        # Initialization parameters.
        init_op = tf.group(tf.global_variables_initializer(), 
                           tf.local_variables_initializer())
        self.session.run(init_op)

        # Restore from ckpt if needed.
        recover_step = 0
        if self.cfg.train.continue_training:
            # For continue training.
            recover_step = self.restore()
        elif self.cfg.checkpoint != '' or (
            len(self.cfg.train.pretrained_scope_list) > 0):
            # For multi-stage training, load pretrained model. 
            # Each trained with given scope.
            self.load_pretrained(self.cfg.model.scope)

        # Load vgg-19 perceptual if needed.
        if self.cfg.loss.perceptual.loss_weight > 0:
            load_perceptual_module(self.session, self.cfg.loss.perceptual)

        # Synch all the nodes for initialization in distributed training.
        if self.is_distributed:
            logger.info(f'Broadcast variables from root rank')
            broadcast_global_variables(self.session, 
                                       self.cfg.env.device, 
                                       self.cfg.env.root_rank)

        # Dump the train graph on the root node.
        if self.world.is_root_rank:
            tf.io.write_graph(self.session.graph_def, 
                              self.cfg.train.output_dir, 
                              'train_graph.pbtxt')
            logger.info(f'Start training.')

        # Train (may continue from recover step)
        self._train(recover_step)

    def prepare_adv_adapt_op(self, d_loss_ops):
        """ 
        Prepare auxiliary ops when adversarial training.
        
        Due to the insufficiency of Ascend platform in dynamic graph which is 
        common in adversarial training, we do the adaptive balance on session 
        run level instead of tf graph. To do this, we define a helper class to
        determine whether to update generator and discriminator each step.

        On adaptive strategy which adjust the training step according to the 
        discriminator loss, we we must collect all the decision, aggregate and 
        sychronize the decision across the nodes.
        
        """
        
        if self.cfg.loss.adversarial.loss_weight > 0. and (
            self.cfg.loss.adversarial.adaptive_strategy):
            logger.info('Using adversarial training with adaptive strategy.')
            logger.info('There will be some warm-start iterations for '
                        'discriminator, while the generator won\'t update.')
            if self.is_distributed:
                # In adaptive strategy, we should manually synchronize the 
                # discriminator losses across the devices.
                logger.info('Distributed adversarial adaptive training. Generating '
                            'synchronize nodes.')
                adv_helper_criteria = allreduce_avg(
                    d_loss_ops['discriminator'],
                    self.cfg.env.device,
                    self.world.rank_size
                    )
            else:
                adv_helper_criteria = d_loss_ops['discriminator']
        else:
            # To unify the interface, define a tf.no_op
            adv_helper_criteria = tf.no_op()
        return adv_helper_criteria

    def prepare_fetches(self):
        """ 
        Prepare watched tensors. In each step, we want to know the 
        generator total loss, each part of generator losses, discriminator
        total loss (if used), and some summary ops.

        Returns:
            g_ops: dict, {op_name: op_tensor} of the generator.
            d_ops: dict, {op_name: op_tensor} of the discriminator.
            losses: dict, {loss_name: loss_tensor} for printing.
            summary_ops: dict, {summary_name: summary_op} for visualization.
            adv_helper_criteria: tensor, the criteria to tell whether update 
                generator or discriminator. May be a hccl operator.
        """
        # prepare train ops, loss ops, summary ops
        g_ops = name_space.get_collection(name_space.GeneratorRunOp)
        g_loss_ops = name_space.get_collection(name_space.GeneratorLoss)

        d_ops = name_space.get_collection(name_space.DiscriminatorRunOp)
        d_loss_ops = name_space.get_collection(name_space.DiscriminatorLoss)

        summary_ops = name_space.get_collection(name_space.Summary)

        adv_helper_criteria = self.prepare_adv_adapt_op(d_loss_ops)

        return g_ops, d_ops, {**g_loss_ops, **d_loss_ops}, summary_ops, adv_helper_criteria

    def prepare_feeds(self):
        """ 
        Prepare feed dict for session run.

        Returns:
            dict, will be fed to session run.
        """
        # TODO: remove learning rate feed dict.
        feed_dict = {self.g_solver.lr: self.g_solver.update_lr()}
        if self.cfg.loss.adversarial.loss_weight > 0:
            feed_dict[self.d_solver.lr] = self.d_solver.update_lr()
        return feed_dict

    def _train(self, init_step=0):
        """ 
        Train steps.

        Args:
            init_step: int, the starting step of training.
        """
        _g_ops, _d_ops, loss_ops, summary_ops, adv_helper_criteria = \
            self.prepare_fetches()
        train_st = time.time()
        for it in range(init_step, self.g_solver.total_step):
            feed_dict = self.prepare_feeds()

            # In adversarial scenario, we use a helper instance to filter 
            # the truly evaluated ops.
            real_g_ops, real_d_ops = self.helper.filter(_g_ops, _d_ops)

            st_time = time.time()
            ops_result, loss_ops_result, adv_helper_criteria_result = \
                self.session.run([{**real_g_ops, **real_d_ops}, 
                                  loss_ops, 
                                  adv_helper_criteria], 
                                  feed_dict=feed_dict)
            once_time = time.time() - st_time

            if self.world.is_root_rank:
                if it > init_step:
                    # Skip the first print_interval steps, whose values 
                    # might be abnormal
                    self.step_time.update(once_time)
                    total_loss = loss_ops_result['loss_total']
                    self.step_loss.update(total_loss)

                if (it + 1) % self.cfg.train.print_interval == 0:
                    self.print(it + 1, ops_result, loss_ops_result)

                if (it + 1) % self.cfg.train.checkpoint_interval == 0:
                    self.save(it + 1)

            # Update adversarial helper function
            self.helper.update_status(adv_helper_criteria_result, it+1)

            # TODO: support tensorboard, summary and evaluation.
            # For tensorboard visualization
            # if self.writer is not None and (it + 1) % 100 == 0:
            #     summary_merge = tf.summary.merge_all()
            #     summary_loss_result = self.session.run(summary_merge, feed_dict=feed_dict)
            #     self.writer.add_summary(summary_loss_result, it + 1)
            
            if (self.cfg.train.dump_intermediate == 'intermediate' 
                and (it + 1) % self.cfg.train.dump_intermediate == 0):
                summary_ops_result = self.session.run(summary_ops)
                # In distributed training, we should run summary ops on all the devices in 
                # order to synchronize. But only the root node will dump the data.
                if self.world.is_root_rank:
                    self.network.dump_summary(it + 1, summary_ops_result)

        train_time = time.time() - train_st
        time_mi = train_time / 60
        logger.info('Training finished. Average step time:{:.2f} ms, total elapse time: {:.1f} min.'
                    .format(np.mean(self.step_time.avg) * 1000, time_mi))
