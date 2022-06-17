# coding: UTF-8
from __future__ import print_function

from easydict import EasyDict as edict
import tensorflow as tf
from npu_bridge.estimator import npu_ops
from example.wdl_outbrain import features

model_cfg = edict()
model_cfg.loss_mode = "batch"
output_op_name = "logits"
wide_loss_op_name = "wide_loss"
deep_loss_op_name = "deep_loss"
metrics_op_name = "metrics"
pred_op_name = "pred"
label_op_name = "label"
deep_var_list = "deep_dense"
wide_var_list = "wide_dense"


class MyModel:
    """
    support performing mean pooling Operation on multi-hot feature.
    dataset_argv: e.g. [10000, 17, [False, ...,True,True], 10]
    """

    def build_model(
            self,
            deep_doc_event_id_emb=None,  # model.tensor_input_list要从build_model的形参里获取
            wide_doc_event_id_emb=None,
            deep_doc_id_emb=None,
            wide_doc_id_emb=None,
            deep_doc_event_source_id_emb=None,
            wide_doc_event_source_id_emb=None,
            deep_event_geo_location_emb=None,
            wide_event_geo_location_emb=None,
            deep_event_country_state_emb=None,
            wide_event_country_state_emb=None,
            deep_doc_event_publisher_id_emb=None,
            wide_doc_event_publisher_id_emb=None,
            deep_event_country_emb=None,
            wide_event_country_emb=None,
            deep_event_hour_emb=None,
            wide_event_hour_emb=None,
            deep_event_platform_emb=None,
            wide_event_platform_emb=None,
            deep_traffic_source_emb=None,
            wide_traffic_source_emb=None,
            deep_event_weekend_emb=None,
            wide_event_weekend_emb=None,
            deep_user_has_already_viewed_doc_emb=None,
            wide_user_has_already_viewed_doc_emb=None,
            deep_doc_event_entity_id_emb=None,
            wide_doc_event_entity_id_emb=None,
            deep_doc_event_topic_id_emb=None,
            wide_doc_event_topic_id_emb=None,
            deep_doc_event_category_id_emb=None,
            wide_doc_event_category_id_emb=None,
            deep_ad_id_emb=None,
            wide_ad_id_emb=None,
            deep_doc_ad_source_id_emb=None,
            wide_doc_ad_source_id_emb=None,
            deep_ad_advertiser_emb=None,
            wide_ad_advertiser_emb=None,
            deep_doc_ad_publisher_id_emb=None,
            wide_doc_ad_publisher_id_emb=None,
            deep_doc_ad_topic_id_emb=None,
            wide_doc_ad_topic_id_emb=None,
            deep_doc_ad_entity_id_emb=None,
            wide_doc_ad_entity_id_emb=None,
            deep_doc_ad_category_id_emb=None,
            wide_doc_ad_category_id_emb=None,
            pop_document_id_conf=None,
            pop_publisher_id_conf=None,
            pop_source_id_conf=None,
            pop_entity_id_conf=None,
            pop_topic_id_conf=None,
            pop_category_id_conf=None,
            pop_document_id_log_01scaled=None,
            pop_publisher_id_log_01scaled=None,
            pop_source_id_log_01scaled=None,
            pop_entity_id_log_01scaled=None,
            pop_topic_id_log_01scaled=None,
            pop_category_id_log_01scaled=None,
            user_views_log_01scaled=None,
            doc_views_log_01scaled=None,
            doc_event_days_since_published_log_01scaled=None,
            doc_event_hour_log_01scaled=None,
            pop_ad_id_conf=None,
            user_doc_ad_sim_categories_conf=None,
            user_doc_ad_sim_topics_conf=None,
            pop_advertiser_id_conf=None,
            pop_campain_id_conf_multipl_log_01scaled=None,
            pop_ad_id_log_01scaled=None,
            pop_advertiser_id_log_01scaled=None,
            pop_campain_id_log_01scaled=None,
            user_doc_ad_sim_categories_log_01scaled=None,
            user_doc_ad_sim_topics_log_01scaled=None,
            user_doc_ad_sim_entities_log_01scaled=None,
            doc_event_doc_ad_sim_categories_log_01scaled=None,
            doc_event_doc_ad_sim_topics_log_01scaled=None,
            doc_event_doc_ad_sim_entities_log_01scaled=None,
            ad_views_log_01scaled=None,
            doc_ad_days_since_published_log_01scaled=None,
            label=None,
            is_training=True):
        print(f'build model start ...')
        with tf.variable_scope("mlp", reuse=tf.AUTO_REUSE):
            self.labels = tf.cast(label, tf.float32)
            cross_layer = False
            batch_norm = False
            self.var_grad = {}
            layer_dims = [1024, 1024, 1024, 1024, 1024]
            act_func = 'relu'

            keep_prob = 1.0
            _lambda = 9e-6
            print(f'deep_doc_event_id_emb: {deep_doc_event_id_emb}')
            print(f'pop_document_id_conf: {pop_document_id_conf}')
            _input_data = {
                'deep_doc_event_id_emb': deep_doc_event_id_emb,
                'wide_doc_event_id_emb': wide_doc_event_id_emb,
                'deep_doc_id_emb': deep_doc_id_emb,
                'wide_doc_id_emb': wide_doc_id_emb,
                'deep_doc_event_source_id_emb': deep_doc_event_source_id_emb,
                'wide_doc_event_source_id_emb': wide_doc_event_source_id_emb,
                'deep_event_geo_location_emb': deep_event_geo_location_emb,
                'wide_event_geo_location_emb': wide_event_geo_location_emb,
                'deep_event_country_state_emb': deep_event_country_state_emb,
                'wide_event_country_state_emb': wide_event_country_state_emb,
                'deep_doc_event_publisher_id_emb': deep_doc_event_publisher_id_emb,
                'wide_doc_event_publisher_id_emb': wide_doc_event_publisher_id_emb,
                'deep_event_country_emb': deep_event_country_emb,
                'wide_event_country_emb': wide_event_country_emb,
                'deep_event_hour_emb': deep_event_hour_emb,
                'wide_event_hour_emb': wide_event_hour_emb,
                'deep_event_platform_emb': deep_event_platform_emb,
                'wide_event_platform_emb': wide_event_platform_emb,
                'deep_traffic_source_emb': deep_traffic_source_emb,
                'wide_traffic_source_emb': wide_traffic_source_emb,
                'deep_event_weekend_emb': deep_event_weekend_emb,
                'wide_event_weekend_emb': wide_event_weekend_emb,
                'deep_user_has_already_viewed_doc_emb': deep_user_has_already_viewed_doc_emb,
                'wide_user_has_already_viewed_doc_emb': wide_user_has_already_viewed_doc_emb,
                'deep_doc_event_entity_id_emb': deep_doc_event_entity_id_emb,
                'wide_doc_event_entity_id_emb': wide_doc_event_entity_id_emb,
                'deep_doc_event_topic_id_emb': deep_doc_event_topic_id_emb,
                'wide_doc_event_topic_id_emb': wide_doc_event_topic_id_emb,
                'deep_doc_event_category_id_emb': deep_doc_event_category_id_emb,
                'wide_doc_event_category_id_emb': wide_doc_event_category_id_emb,
                'pop_document_id_conf': pop_document_id_conf,
                'pop_publisher_id_conf': pop_publisher_id_conf,
                'pop_source_id_conf': pop_source_id_conf,
                'pop_entity_id_conf': pop_entity_id_conf,
                'pop_topic_id_conf': pop_topic_id_conf,
                'pop_category_id_conf': pop_category_id_conf,
                'pop_document_id_log_01scaled': pop_document_id_log_01scaled,
                'pop_publisher_id_log_01scaled': pop_publisher_id_log_01scaled,
                'pop_source_id_log_01scaled': pop_source_id_log_01scaled,
                'pop_entity_id_log_01scaled': pop_entity_id_log_01scaled,
                'pop_topic_id_log_01scaled': pop_topic_id_log_01scaled,
                'pop_category_id_log_01scaled': pop_category_id_log_01scaled,
                'user_views_log_01scaled': user_views_log_01scaled,
                'doc_views_log_01scaled': doc_views_log_01scaled,
                'doc_event_days_since_published_log_01scaled': doc_event_days_since_published_log_01scaled,
                'doc_event_hour_log_01scaled': doc_event_hour_log_01scaled,
                'deep_ad_id_emb': deep_ad_id_emb,
                'wide_ad_id_emb': wide_ad_id_emb,
                'deep_doc_ad_source_id_emb': deep_doc_ad_source_id_emb,
                'wide_doc_ad_source_id_emb': wide_doc_ad_source_id_emb,
                'deep_ad_advertiser_emb': deep_ad_advertiser_emb,
                'wide_ad_advertiser_emb': wide_ad_advertiser_emb,
                'deep_doc_ad_publisher_id_emb': deep_doc_ad_publisher_id_emb,
                'wide_doc_ad_publisher_id_emb': wide_doc_ad_publisher_id_emb,
                'deep_doc_ad_topic_id_emb': deep_doc_ad_topic_id_emb,
                'wide_doc_ad_topic_id_emb': wide_doc_ad_topic_id_emb,
                'deep_doc_ad_entity_id_emb': deep_doc_ad_entity_id_emb,
                'wide_doc_ad_entity_id_emb': wide_doc_ad_entity_id_emb,
                'deep_doc_ad_category_id_emb': deep_doc_ad_category_id_emb,
                'wide_doc_ad_category_id_emb': wide_doc_ad_category_id_emb,
                'pop_ad_id_conf': pop_ad_id_conf,
                'user_doc_ad_sim_categories_conf': user_doc_ad_sim_categories_conf,
                'user_doc_ad_sim_topics_conf': user_doc_ad_sim_topics_conf,
                'pop_advertiser_id_conf': pop_advertiser_id_conf,
                'pop_campain_id_conf_multipl_log_01scaled': pop_campain_id_conf_multipl_log_01scaled,
                'pop_ad_id_log_01scaled': pop_ad_id_log_01scaled,
                'pop_advertiser_id_log_01scaled': pop_advertiser_id_log_01scaled,
                'pop_campain_id_log_01scaled': pop_campain_id_log_01scaled,
                'user_doc_ad_sim_categories_log_01scaled': user_doc_ad_sim_categories_log_01scaled,
                'user_doc_ad_sim_topics_log_01scaled': user_doc_ad_sim_topics_log_01scaled,
                'user_doc_ad_sim_entities_log_01scaled': user_doc_ad_sim_entities_log_01scaled,
                'doc_event_doc_ad_sim_categories_log_01scaled': doc_event_doc_ad_sim_categories_log_01scaled,
                'doc_event_doc_ad_sim_topics_log_01scaled': doc_event_doc_ad_sim_topics_log_01scaled,
                'doc_event_doc_ad_sim_entities_log_01scaled': doc_event_doc_ad_sim_entities_log_01scaled,
                'ad_views_log_01scaled': ad_views_log_01scaled,
                'doc_ad_days_since_published_log_01scaled': doc_ad_days_since_published_log_01scaled}

            wide_inputs, deep_inputs = features.get_feature_inputs(
                _input_data)
            self.wide_layer = tf.transpose(
                tf.concat([tf.transpose(x) for x in wide_inputs], axis=0))
            self.deep_layer = tf.transpose(
                tf.concat([tf.transpose(x) for x in deep_inputs], axis=0))

            self.embed_dim = 2784
            self.all_layer_dims = [self.embed_dim] + layer_dims + [1]
            self.log = (
                'embedding layer: %d\nlayers: %s\nactivate: %s\n'
                'keep_prob: %g\nl2(lambda): %g\n' %
                (self.embed_dim, self.all_layer_dims, act_func, keep_prob, _lambda))

            with tf.variable_scope("wide_embeddings", reuse=tf.AUTO_REUSE):
                self.wide_b = tf.get_variable(
                    'wide_b',
                    [1],
                    initializer=tf.random_uniform_initializer(
                        -0.01,
                        0.01),
                    dtype=tf.float32,
                    collections=[
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        "wide",
                        "wide_bias"])

            with tf.variable_scope("mlp", reuse=tf.AUTO_REUSE):
                self.h_w, self.h_b = [], []
                for i in range(len(self.all_layer_dims) - 1):
                    self.h_w.append(tf.get_variable('h%d_w' % (i + 1),
                                                    shape=self.all_layer_dims[i: i + 2],
                                                    initializer=tf.random_uniform_initializer(-0.01,
                                                                                              0.01),
                                                    dtype=tf.float32,
                                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                 "deep",
                                                                 "mlp_wts"]))
                    self.h_b.append(tf.get_variable('h%d_b' % (i + 1),
                                                    shape=[self.all_layer_dims[i + 1]],
                                                    initializer=tf.zeros_initializer,
                                                    dtype=tf.float32,
                                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                 "deep",
                                                                 "mlp_bias"]))

            wideout = self.wide_forward(self.wide_layer)
            y = self.forward(
                self.deep_layer, act_func, keep_prob, training=is_training,
                cross_layer=cross_layer, batch_norm=batch_norm)
            y = y + wideout
            self.train_preds = tf.sigmoid(y, name='predicitons')

            basic_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y, labels=self.labels)

            self.wide_loss = tf.reduce_mean(basic_loss)
            self.deep_loss = tf.reduce_mean(
                basic_loss)  # + _lambda * tf.nn.l2_loss(embedding)

            self.l2_loss = tf.constant([0])  # self.loss
            self.log_loss = basic_loss

        print(f'build model end ...')
        return {output_op_name: y,
                wide_loss_op_name: self.wide_loss,
                deep_loss_op_name: self.deep_loss,
                metrics_op_name: basic_loss,
                pred_op_name: self.train_preds,
                label_op_name: self.labels,
                deep_var_list: tf.get_collection('deep'),
                wide_var_list: tf.get_collection('wide')}

    def wide_forward(self, wide_part):
        wide_output = tf.reshape(
            (tf.reduce_sum(wide_part, axis=1) + self.wide_b), shape=[-1, ], name="wide_out")
        return wide_output

    def forward(self, deep_part, act_func, keep_prob,
                training, cross_layer=False, batch_norm=False):
        hidden_output = tf.reshape(deep_part, [-1, self.embed_dim])
        cross_layer_output = None
        for i in range(len(self.h_w)):
            if training:
                hidden_output = tf.matmul(
                    npu_ops.dropout(
                        self.activate(
                            act_func,
                            hidden_output),
                        keep_prob=keep_prob),
                    self.h_w[i])
            else:
                hidden_output = tf.matmul(
                    self.activate(
                        act_func,
                        hidden_output),
                    self.h_w[i])
            hidden_output = hidden_output + self.h_b[i]

            if batch_norm:
                hidden_output = tf.layers.batch_normalization(
                    hidden_output, training=training)
            if cross_layer_output is not None:
                cross_layer_output = tf.concat(
                    [cross_layer_output, hidden_output], 1)
            else:
                cross_layer_output = hidden_output

            if cross_layer and i == len(self.h_w) - 2:
                hidden_output = cross_layer_output
        return tf.reshape(hidden_output, [-1, ])

    def activate(self, act_func, x):
        if act_func == 'tanh':
            return tf.tanh(x)
        elif act_func == 'relu':
            return tf.nn.relu(x)
        else:
            return tf.sigmoid(x)


my_model = MyModel()
