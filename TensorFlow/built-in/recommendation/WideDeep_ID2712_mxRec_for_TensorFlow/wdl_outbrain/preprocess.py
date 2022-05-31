import tensorflow as tf


def my_preprocess(batch):
    tensor_name_list = [
        'doc_event_id',
        'doc_id',
        'doc_event_source_id',
        'event_geo_location',
        'event_country_state',
        'doc_event_publisher_id',
        'event_country',
        'event_hour',
        'event_platform',
        'traffic_source',
        'event_weekend',
        'user_has_already_viewed_doc',
        'doc_event_entity_id',
        'doc_event_topic_id',
        'doc_event_category_id',
        'ad_id',
        'doc_ad_source_id',
        'ad_advertiser',
        'doc_ad_publisher_id',
        'doc_ad_topic_id',
        'doc_ad_entity_id',
        'doc_ad_category_id']
    voc_size_list = [
        300000,
        100000,
        4000,
        2500,
        2000,
        1000,
        300,
        6,
        3,
        3,
        2,
        2,
        10000,
        350,
        100,
        250000,
        4000,
        2500,
        1000,
        350,
        10000,
        100]
    for tensor_name, voc_size in zip(tensor_name_list, voc_size_list):
        batch[tensor_name] = tf.math.floormod(batch[tensor_name], voc_size)
    return batch
