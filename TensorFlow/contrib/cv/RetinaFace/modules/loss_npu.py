import tensorflow as tf


# def _smooth_l1_loss(y_true, y_pred):
#     t = tf.abs(y_pred - y_true)
#     return tf.where(t < 1, 0.5 * t ** 2, t - 0.5)

def _snowball_smooth_l1_loss(y_true, y_pred, bool_mask, clip_value=None):
    # y_pred = _check_nan(y_pred, 0)
    # y_true = _check_nan(y_true, y_pred)
    # if clip_value is not None:
    #     y_pred = tf.clip_by_value(y_pred, -1.0*clip_value, 1.0*clip_value)
    float_mask = tf.cast(bool_mask, dtype=tf.float32)
    true_num = tf.count_nonzero(float_mask)
    float_true = tf.multiply(y_true, float_mask)
    float_pred = tf.multiply(y_pred, float_mask)
    diff = tf.abs(float_pred - float_true)
    loss_ = tf.where(diff < 1, 0.5 * (diff * diff), diff - 0.5)
    loss = tf.cond(true_num > 0, lambda: tf.reduce_sum(loss_)/tf.cast(true_num, dtype=tf.float32), lambda: 0*tf.reduce_sum(loss_))

    return loss


def _snowball_crossentry(y_true=None, y_pred=None, axis=-1):
    # scale preds so that the class probas of each sample sum to 1
    s_y_pred = tf.math.l2_normalize(y_pred, axis=axis, epsilon=1e-6)
    # s_y_pred = y_pred/(tf.reduce_sum(y_pred, axis, True)+1e-12)
    epsilon_ = tf.constant(1e-7, y_pred.dtype.base_dtype)
    s_y_pred = tf.clip_by_value(s_y_pred, epsilon_, 1. - epsilon_)
    result = -tf.reduce_sum(y_true * tf.log(s_y_pred), axis, True)

    return result


def _check_nan(v, max_v):
    max_v = tf.ones_like(v)*max_v if not isinstance(max_v, type(v)) else max_v
    tmp_v = tf.where(tf.is_nan(v), max_v, v)
    return v


def MultiBoxLoss(num_class=2, neg_pos_ratio=3):
    """multi-box loss"""
    def multi_box_loss(y_true, y_pred):
        num_batch = tf.shape(y_true)[0]
        num_prior = tf.shape(y_true)[1]

        loc_pred = tf.reshape(y_pred[0], [num_batch * num_prior, 4])
        landm_pred = tf.reshape(y_pred[1], [num_batch * num_prior, 10])
        class_pred = tf.reshape(y_pred[2], [num_batch * num_prior, num_class])
        loc_true = tf.reshape(y_true[..., :4], [num_batch * num_prior, 4])
        landm_true = tf.reshape(y_true[..., 4:14], [num_batch * num_prior, 10])
        landm_valid = tf.reshape(y_true[..., 14], [num_batch * num_prior, 1])
        class_true = tf.reshape(y_true[..., 15], [num_batch * num_prior, 1])

        # define filter mask: class_true = 1 (pos), 0 (neg), -1 (ignore)
        #                     landm_valid = 1 (w landm), 0 (w/o landm)
        mask_pos = tf.equal(class_true, 1)
        mask_neg = tf.equal(class_true, 0)
        mask_landm = tf.logical_and(tf.equal(landm_valid, 1), mask_pos)

        # landm loss (smooth L1)
        mask_landm_b = tf.broadcast_to(mask_landm, tf.shape(landm_true))
        loss_landm = _snowball_smooth_l1_loss(landm_true, landm_pred, mask_landm_b)

        # localization loss (smooth L1)
        mask_pos_b = tf.broadcast_to(mask_pos, tf.shape(loc_true))
        loss_loc = _snowball_smooth_l1_loss(loc_true, loc_pred, mask_pos_b, clip_value=2)

        # classification loss (crossentropy)
        # 1. compute max conf across batch for hard negative mining
        loss_class = tf.where_v2(mask_neg, 1 - tf.expand_dims(class_pred[:, 0], -1), 0)

        # 2. hard negative mining
        loss_class = tf.reshape(loss_class, [num_batch, num_prior])
        loss_class_idx = tf.argsort(loss_class, axis=1, direction='DESCENDING')
        loss_class_idx_rank = tf.argsort(loss_class_idx, axis=1)
        mask_pos_per_batch = tf.reshape(mask_pos, [num_batch, num_prior])
        num_pos_per_batch = tf.reduce_sum(tf.cast(mask_pos_per_batch, tf.float32), 1, keepdims=True)
        num_pos_per_batch = tf.maximum(num_pos_per_batch, 1)
        num_neg_per_batch = tf.minimum(neg_pos_ratio * num_pos_per_batch, tf.cast(num_prior, tf.float32) - 1)
        mask_hard_neg = tf.reshape(tf.cast(loss_class_idx_rank, tf.float32) < num_neg_per_batch, [num_batch * num_prior, 1])

        # 3. classification loss including positive and negative examples
        loss_class_mask = tf.logical_or(mask_pos, mask_hard_neg)
        float_loss_class_mask = tf.cast(loss_class_mask, dtype=tf.float32)
        true_num = tf.count_nonzero(float_loss_class_mask)
        one_hot_label = tf.concat([tf.cast(mask_neg, dtype=tf.float32), tf.cast(mask_pos, dtype=tf.float32)], -1)
        all_loss_value = _snowball_crossentry(y_true=one_hot_label, y_pred=class_pred)
        selected_loss_value = float_loss_class_mask*all_loss_value
        loss_class = tf.cond(true_num > 0, lambda: tf.reduce_sum(selected_loss_value)/tf.cast(true_num, dtype=tf.float32), lambda: 0*tf.reduce_sum(selected_loss_value))

        return loss_loc, loss_landm, loss_class

    return multi_box_loss
