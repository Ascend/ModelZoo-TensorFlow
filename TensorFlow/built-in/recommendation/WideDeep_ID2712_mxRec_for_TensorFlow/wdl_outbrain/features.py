import tensorflow as tf
import numpy as np


INT_COLUMNS = [
    'user_views',
    'ad_views',
    'doc_views',
    'doc_event_days_since_published',
    'doc_event_hour',
    'doc_ad_days_since_published']

FLOAT_COLUMNS_LOG_BIN_TRANSFORM = [
    'pop_ad_id',
    'pop_ad_id_conf_multipl',
    'pop_document_id',
    'pop_document_id_conf_multipl',
    'pop_publisher_id',
    'pop_publisher_id_conf_multipl',
    'pop_advertiser_id',
    'pop_advertiser_id_conf_multipl',
    'pop_campain_id',
    'pop_campain_id_conf_multipl',
    'pop_doc_event_doc_ad',
    'pop_doc_event_doc_ad_conf_multipl',
    'pop_source_id',
    'pop_source_id_conf_multipl',
    'pop_source_id_country',
    'pop_source_id_country_conf_multipl',
    'pop_entity_id',
    'pop_entity_id_conf_multipl',
    'pop_entity_id_country',
    'pop_entity_id_country_conf_multipl',
    'pop_topic_id',
    'pop_topic_id_conf_multipl',
    'pop_topic_id_country',
    'pop_topic_id_country_conf_multipl',
    'pop_category_id',
    'pop_category_id_conf_multipl',
    'pop_category_id_country',
    'pop_category_id_country_conf_multipl',
    'user_doc_ad_sim_categories',
    'user_doc_ad_sim_categories_conf_multipl',
    'user_doc_ad_sim_topics',
    'user_doc_ad_sim_topics_conf_multipl',
    'user_doc_ad_sim_entities',
    'user_doc_ad_sim_entities_conf_multipl',
    'doc_event_doc_ad_sim_categories',
    'doc_event_doc_ad_sim_categories_conf_multipl',
    'doc_event_doc_ad_sim_topics',
    'doc_event_doc_ad_sim_topics_conf_multipl',
    'doc_event_doc_ad_sim_entities',
    'doc_event_doc_ad_sim_entities_conf_multipl']

FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM = [
    'pop_ad_id_conf',
    'pop_document_id_conf',
    'pop_publisher_id_conf',
    'pop_advertiser_id_conf',
    'pop_campain_id_conf',
    'pop_doc_event_doc_ad_conf',
    'pop_source_id_conf',
    'pop_source_id_country_conf',
    'pop_entity_id_conf',
    'pop_entity_id_country_conf',
    'pop_topic_id_conf',
    'pop_topic_id_country_conf',
    'pop_category_id_conf',
    'pop_category_id_country_conf',
    'user_doc_ad_sim_categories_conf',
    'user_doc_ad_sim_topics_conf',
    'user_doc_ad_sim_entities_conf',
    'doc_event_doc_ad_sim_categories_conf',
    'doc_event_doc_ad_sim_topics_conf',
    'doc_event_doc_ad_sim_entities_conf']


# Let's define the columns we're actually going to use
# during training
REQUEST_SINGLE_HOT_COLUMNS = [
    "doc_event_id",
    "doc_id",
    "doc_event_source_id",
    "event_geo_location",
    "event_country_state",
    "doc_event_publisher_id",
    "event_country",
    "event_hour",
    "event_platform",
    "traffic_source",
    "event_weekend",
    "user_has_already_viewed_doc"]

REQUEST_MULTI_HOT_COLUMNS = [
    "doc_event_entity_id",
    "doc_event_topic_id",
    "doc_event_category_id"]

REQUEST_NUMERIC_COLUMNS = [
    "pop_document_id_conf",
    "pop_publisher_id_conf",
    "pop_source_id_conf",
    "pop_entity_id_conf",
    "pop_topic_id_conf",
    "pop_category_id_conf",
    "pop_document_id",
    "pop_publisher_id",
    "pop_source_id",
    "pop_entity_id",
    "pop_topic_id",
    "pop_category_id",
    "user_views",
    "doc_views",
    "doc_event_days_since_published",
    "doc_event_hour"]

ITEM_SINGLE_HOT_COLUMNS = [
    "ad_id",
    "doc_ad_source_id",
    "ad_advertiser",
    "doc_ad_publisher_id"]

ITEM_MULTI_HOT_COLUMNS = [
    "doc_ad_topic_id",
    "doc_ad_entity_id",
    "doc_ad_category_id"]

ITEM_NUMERIC_COLUMNS = [
    "pop_ad_id_conf",
    "user_doc_ad_sim_categories_conf",
    "user_doc_ad_sim_topics_conf",
    "pop_advertiser_id_conf",
    "pop_campain_id_conf_multipl",
    "pop_ad_id",
    "pop_advertiser_id",
    "pop_campain_id",
    "user_doc_ad_sim_categories",
    "user_doc_ad_sim_topics",
    "user_doc_ad_sim_entities",
    "doc_event_doc_ad_sim_categories",
    "doc_event_doc_ad_sim_topics",
    "doc_event_doc_ad_sim_entities",
    "ad_views",
    "doc_ad_days_since_published"]

TRAINING_COLUMNS = (
    REQUEST_SINGLE_HOT_COLUMNS +
    REQUEST_MULTI_HOT_COLUMNS +
    REQUEST_NUMERIC_COLUMNS +
    ITEM_SINGLE_HOT_COLUMNS +
    ITEM_MULTI_HOT_COLUMNS +
    ITEM_NUMERIC_COLUMNS)

HASH_BUCKET_SIZES = {
    'doc_event_id': 300000,
    'ad_id': 250000,
    'doc_id': 100000,
    'doc_ad_entity_id': 10000,
    'doc_event_entity_id': 10000,
    'doc_ad_source_id': 4000,
    'doc_event_source_id': 4000,
    'event_geo_location': 2500,
    'ad_advertiser': 2500,
    'event_country_state': 2000,
    'doc_ad_publisher_id': 1000,
    'doc_event_publisher_id': 1000,
    'doc_ad_topic_id': 350,
    'doc_event_topic_id': 350,
    'event_country': 300,
    'doc_ad_category_id': 100,
    'doc_event_category_id': 100}

IDENTITY_NUM_BUCKETS = {
    'event_hour': 6,
    'event_platform': 3,
    'traffic_source': 3,
    'event_weekend': 2,
    'user_has_already_viewed_doc': 2}


def get_feature_inputs(input_data=None):
    training_columns = TRAINING_COLUMNS

    tf.compat.v1.logging.warn(
        'number of features: {}'.format(
            len(training_columns)))

    wide_inputs = []
    deep_inputs = []
    for column_name in training_columns:
        if column_name in HASH_BUCKET_SIZES or column_name in IDENTITY_NUM_BUCKETS:
            deep_emb_name = 'deep_{}_emb'.format(column_name)
            deep_emb_res = input_data.get(deep_emb_name)
            deep_emb_shape = deep_emb_res.get_shape().as_list()
            if len(deep_emb_shape) < 1:
                print(f'deep_{column_name}_emb shape is {deep_emb_shape}')
                raise ValueError
            deep_inputs.append(tf.reshape(deep_emb_res,
                                          [-1, np.prod(deep_emb_shape[1:])]))
            wide_emb_name = 'wide_{}_emb'.format(column_name)
            wide_emb_res = input_data.get(wide_emb_name)
            wide_emb_shape = wide_emb_res.get_shape().as_list()
            if len(wide_emb_shape) < 1:
                raise ValueError
            wide_inputs.append(tf.reshape(wide_emb_res,
                                          [-1, np.prod(wide_emb_shape[1:])]))
        elif column_name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
            deep_inputs.append(input_data.get(column_name))
            with tf.variable_scope("wide_embeddings", reuse=tf.AUTO_REUSE):
                wide_emb = tf.get_variable(
                    'wide_' + column_name,
                    shape=[1],
                    initializer=tf.random_uniform_initializer(
                        -0.01,
                        0.01),
                    dtype=tf.float32,
                    collections=[
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        "wide",
                        "wide_emb"])
            wide_inputs.append(
                tf.ones_like(
                    input_data.get(column_name),
                    dtype=tf.float32) *
                wide_emb)
        elif column_name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM or column_name in INT_COLUMNS:
            deep_inputs.append(input_data.get(column_name + '_log_01scaled'))
            with tf.variable_scope("wide_embeddings", reuse=tf.AUTO_REUSE):
                wide_emb = tf.get_variable(
                    'wide_' + column_name,
                    shape=[1],
                    initializer=tf.random_uniform_initializer(
                        -0.01,
                        0.01),
                    dtype=tf.float32,
                    collections=[
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        "wide",
                        "wide_emb"])
            wide_inputs.append(
                tf.ones_like(
                    input_data.get(
                        column_name +
                        '_log_01scaled'),
                    dtype=tf.float32) *
                wide_emb)
    tf.compat.v1.logging.warn('deep inputs: {}'.format(len(deep_inputs)))
    tf.compat.v1.logging.warn('wide inputs: {}'.format(len(wide_inputs)))
    return wide_inputs, deep_inputs
