import numpy as np
import scipy as sci
import tensorflow as tf
import packing.packing_fea as packing_fea

from tensorflow.python.training import moving_averages
from packing.packing_env import PackingEnv
from gym import spaces
from stable_baselines.common.distributions import make_proba_dist_type


# Batch_norm adapted from:
#   https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/ops.py
#   https://github.com/tensorflow/models/blob/master/research/inception/inception/inception_train.py
#   https://stackoverflow.com/questions/41819080/how-do-i-use-batch-normalization-in-a-multi-gpu-setting-in-tensorflow
# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'
def batch_norm(inp,
               is_training,
               center=True,
               scale=True,
               epsilon=0.001,
               decay=0.99,
               name=None,
               reuse=None):

    """Adds a Batch Normalization layer.

    Args:
        inp: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
        is_training: whether or not the model is in training mode.
        center: If True, subtract beta. If False, beta is not created and
            ignored.
        scale: If True, multiply by gamma. If False, gamma is
            not used. When the next layer is linear (also e.g. ReLU), this can be
            disabled since the scaling can be done by the next layer.
        epsilon: small float added to variance to avoid dividing by zero.
        decay: decay for the moving average.
        name: Optional scope for variable_scope.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.

    Returns:
        a tensor representing the output of the operation.
    """

    if name == None:
        name = "batch_norm"

    inputs_shape = inp.get_shape()
    with tf.variable_scope(name, reuse=reuse):
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = tf.get_variable(
                'beta',
                shape=params_shape,
                initializer=tf.zeros_initializer(),
                trainable=True)
        if scale:
            gamma = tf.get_variable(
                'gamma',
                shape=params_shape,
                initializer=tf.ones_initializer(),
                trainable=True)

        moving_mean = tf.get_variable(
            'moving_mean',
            params_shape,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance',
            params_shape,
            initializer=tf.ones_initializer(),
            trainable=False)

        def mean_var_from_data():
            # Calculate the moments based on the individual batch.
            mean, variance = tf.nn.moments(inp, axis)
            return mean, variance

        mean, variance = tf.cond(
                        pred=is_training,
                        true_fn=mean_var_from_data,
                        false_fn=lambda: (moving_mean, moving_variance))

        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, decay)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, decay)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    # Normalize the activations.
    outputs = tf.nn.batch_normalization(inp, mean, variance, beta, gamma,
                                        epsilon)

    return outputs


def residual3d(inp, is_training, relu_after=True, add_bn=True,
               name=None, reuse=None):
    """ 3d equivalent to 2d residual layer

        Args:
            inp (tensor[batch_size, d, h, w, channels]):
            is_training (tensor[bool]):
            relu_after (bool):
            add_bn (bool): add bn before every relu
            name (string):
            reuse (bool):
    """

    if name == None:
        name = "residual3d"

    out_dim = (int)(inp.shape[-1])
    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.layers.conv3d(
            inp, filters=out_dim, kernel_size=[3, 3, 3],
            strides=[1, 1, 1], padding="same", activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(), name="layer1",
            reuse=reuse)

        if add_bn:
            out1 = batch_norm(
                inp=out1,
                is_training=is_training,
                name="norm1",
                reuse=reuse)

        out1 = tf.nn.relu(out1)

        out2 = tf.layers.conv3d(
            out1, filters=out_dim, kernel_size=[3, 3, 3],
            strides=[1, 1, 1], padding="same", activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(), name="layer2",
            reuse=reuse)

        if relu_after and add_bn:
            out2 = batch_norm(
                inp=out2,
                is_training=is_training,
                name="norm2",
                reuse=reuse)

    if relu_after:
        return tf.nn.relu(inp + out2)
    else:
        return inp + out2


def proj_residual3d(inp, is_training, relu_after=True, add_bn=True,
                    name=None, reuse=None):
    """ 3d equivalent to 2d residual projection layer

        Args:
            inp (tensor[batch_size, d, h, w, channels]):
            is_training (tensor[bool]):
            relu_after (bool):
            add_bn (bool): add bn before every relu
            name (string):
            reuse (bool):
    """

    if name == None:
        name = "proj_residual3d"

    out_dim = (int)(inp.shape[-1]) * 2
    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.layers.conv3d(
            inp, filters=out_dim, kernel_size=[3, 3, 3],
            strides=[2, 2, 2], padding="same", activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(), name="layer1",
            reuse=reuse)

        if add_bn:
            out1 = batch_norm(
                inp=out1,
                is_training=is_training,
                name="norm1",
                reuse=reuse)

        out1 = tf.nn.relu(out1)

        out2 = tf.layers.conv3d(
            out1, filters=out_dim, kernel_size=[3, 3, 3],
            strides=[1, 1, 1], padding="same", activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(), name="layer2",
            reuse=reuse)

        if relu_after and add_bn:
            out2 = batch_norm(
                inp=out2,
                is_training=is_training,
                name="norm2",
                reuse=reuse)

        proj_out1 = tf.layers.conv3d(
            inp, filters=out_dim, kernel_size=[3, 3, 3],
            strides=[2, 2, 2], padding="same", activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(), name="layer3",
            reuse=reuse)

    if relu_after:
        return tf.nn.relu(proj_out1 + out2)
    else:
        return proj_out1 + out2


def encoder3d(inp, is_training, add_bn, feature_size, name=None, reuse=None):
    """ Extracts features from the inp by doing 3d convolutions

        Note that there is no activation at the last layer
        For h=d=w > 25, we first project them to 25 using projection layers

        Args:
            inp (tensor[batch_size, d, h, w, channels] or
                   tensor[batch_size, d, h, w, channels]):
            is_training(bool)
            feature_size (int):
            name (string):
            reuse (bool):

        Return:
            features (tensor[batch_size, feature_size])

        Assumptions:
            h=d=w
    """

    assert len(inp.shape) in [4, 5]
    if len(inp.shape) == 4:
        inp = tf.expand_dims(inp, 4)

    if name == None:
        name = "encoder3d"

    out_dim = feature_size / 8
    num_intial_proj = int(np.log2((int)(inp.shape[1]) / 25))
    with tf.variable_scope(name, reuse=reuse):

        for i in range(num_intial_proj):
            inp = proj_residual3d(
                inp=inp,
                is_training=is_training,
                relu_after=True,
                add_bn=add_bn,
                name="initial_proj_{}".format(i),
                reuse=reuse)

        out1 = tf.layers.conv3d(
            inp, filters=out_dim, kernel_size=[3, 3, 3],
            strides=[1, 1, 1], padding="same", activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(), name="out1",
            reuse=reuse)

        if add_bn:
            out1 = batch_norm(
                inp=out1,
                is_training=is_training,
                name="norm1",
                reuse=reuse)

        out1 = tf.nn.relu(out1)

        proj_res1 = proj_residual3d(
            inp=out1,
            is_training=is_training,
            relu_after=True,
            add_bn=add_bn,
            name="proj_res1",
            reuse=reuse)

        res1 = residual3d(
            inp=proj_res1,
            is_training=is_training,
            relu_after=True,
            add_bn=add_bn,
            name="res1",
            reuse=reuse)

        proj_res2 = proj_residual3d(
            inp=res1,
            is_training=is_training,
            relu_after=True,
            add_bn=add_bn,
            name="proj_res2",
            reuse=reuse)

        res2 = residual3d(
            inp=proj_res2,
            is_training=is_training,
            relu_after=True,
            add_bn=add_bn,
            name="res2",
            reuse=reuse)

        proj_res3 = proj_residual3d(
            inp=res2,
            is_training=is_training,
            relu_after=True,
            add_bn=add_bn,
            name="proj_res3",
            reuse=reuse)

        res3 = residual3d(
            inp=proj_res3,
            is_training=is_training,
            relu_after=False,
            add_bn=add_bn,
            name="res3",
            reuse=reuse)

    depth = int(res3.shape[1])
    height = int(res3.shape[2])
    width = int(res3.shape[3])
    features = tf.layers.average_pooling3d(res3, [depth, height, width],
                                           strides=[1, 1, 1], padding="valid")
    features = tf.squeeze(features, [1, 2, 3])

    return features


def residual_fc(
    inp,
    is_training,
    relu_after=True,
    add_bn=True,
    name=None,
    reuse=None):

    """ Returns a residual block fc layer """
    if name == None:
        name = "residual_fc"

    inp_dim = int(inp.shape[-1])
    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=inp_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_1", reuse=reuse)
        if add_bn:
            out1 = batch_norm(
                inp=out1,
                is_training=is_training,
                name="norm_1",
                reuse=reuse)

        out1 = tf.nn.relu(out1)

        out2 = tf.contrib.layers.fully_connected(
                    out1,
                    num_outputs=inp_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_2", reuse=reuse)

        if relu_after and add_bn:
            out2 = batch_norm(
                inp=out2,
                is_training=is_training,
                name="norm_2",
                reuse=reuse)

        if relu_after:
            out = tf.nn.relu(inp + out2)
        else:
            out = inp + out2

    return out


def proj_residual_fc(
    inp,
    is_training,
    out_dim,
    relu_after=True,
    add_bn=True,
    name=None,
    reuse=None):

    """ Returns a residual block fc layer with projection """
    if name == None:
        name = "proj_residual_fc"

    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_1", reuse=reuse)
        if add_bn:
            out1 = batch_norm(
                inp=out1,
                is_training=is_training,
                name="norm_1",
                reuse=reuse)

        out1 = tf.nn.relu(out1)

        out2 = tf.contrib.layers.fully_connected(
                    out1,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_2", reuse=reuse)

        if relu_after and add_bn:
            out2 = batch_norm(
                inp=out2,
                is_training=is_training,
                name="norm_2",
                reuse=reuse)

        out3 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_3", reuse=reuse)

        if relu_after:
            out = tf.nn.relu(out3 + out2)
        else:
            out = out3 + out2

    return out


def final_fc_layers(
    inp,
    out_dim,
    name=None,
    reuse=None):
    """ These are to be used before predicting the logits. As Hei suggested,
        the fc layers of the last layer should not have bn so no bn capability.
    """

    if name == None:
        name = "final_fc_layers"

    inp_dim = int(inp.shape[-1])
    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=inp_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_1", reuse=reuse)

        out2 = tf.contrib.layers.fully_connected(
                    out1,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_2", reuse=reuse)

    return out2


def ser_fc_layers(
    inp,
    is_training,
    num_hid,
    hid_dim,
    out_dim,
    relu_after=False,
    add_bn=True,
    name=None,
    reuse=None):
    """ Returns a series of fully connected layer

        Here, each hidden layer means a residual block. The first fc layer maps
        the inp to hid_dim. Then, a series of (num_hid-1) fc layers. Finally, a
        fc layer which maps to out_dim.  There is a relu in all the hidden
        dimension. The presense of the last activation is controlled using
        relu_after. If add_bn, a bn layer added before every relu
    """

    if name == None:
        name = "ser_fc_layer"

    with tf.variable_scope(name, reuse=reuse):
        if num_hid > 0:
            if int(inp.shape[-1]) == hid_dim:
                inp = residual_fc(
                    inp=inp,
                    is_training=is_training,
                    relu_after=True,
                    add_bn=add_bn,
                    name="layer_0",
                    reuse=reuse)
            else:
                inp = proj_residual_fc(
                    inp=inp,
                    is_training=is_training,
                    out_dim=hid_dim,
                    relu_after=True,
                    add_bn=add_bn,
                    name="layer_0",
                    reuse=reuse)

            for i in range(num_hid - 1):
                inp = residual_fc(
                    inp=inp,
                    is_training=is_training,
                    relu_after=True,
                    add_bn=add_bn,
                    name="layer_{}".format(i + 1),
                    reuse=reuse)

        if hid_dim == out_dim:
            out = residual_fc(
                inp=inp,
                is_training=is_training,
                relu_after=relu_after,
                add_bn=add_bn,
                name="layer_last",
                reuse=reuse)
        else:
            out = proj_residual_fc(
                inp=inp,
                is_training=is_training,
                out_dim=out_dim,
                relu_after=relu_after,
                add_bn=add_bn,
                name="layer_last",
                reuse=reuse)

    return out


def get_col_sha_enc(sha_enc):
    """ Returns max-pooled features from all the shapes

        Args:
            sha_enc (tensor[batch_size, num_shapes, features]):
    """

    _sha_enc = tf.expand_dims(sha_enc, axis=3)
    _col_sha_enc = tf.nn.max_pool(
        _sha_enc, ksize=[1, PackingEnv.MAX_NUM_SHA, 1, 1],
        strides=[1, 1, 1, 1], padding="VALID")
    col_sha_enc = tf.squeeze(_col_sha_enc, [1, 3])

    return col_sha_enc


def masked_softmax_entropy_loss(logits, mask, ground_truth,
                                weight_sample, sum_weight_sample=False):
    """
        A numerically stable entropy, softmax and cross entropy loss
        calculater that works even in the case of masking some logits

        TODO: combine capability to apply both weighting simultaneously

        Args:
            logits (tensor[batch_size, features]):
            mask (tensor[batch_size, features]):
            groud_truth (tensor[batch_size, features]): one hot, ground_truth can
                be 1 at multiple location to signify multiple possible actions
            weight_sample (tensor[batch_size]):
            sum_weight_sample (bool): each sample weighted by 1 / log(mask_sum),
                when true, weight sample must be [1.0, 1.0, 1.0] i.e equal weighting

        Returns:
            softmax (tensor[batch_size, features]):
            entropy (tensor[batch_size, 1]):
            cross_entropy_loss (tensor[]):
    """

    _max_logits = tf.reduce_max(logits * mask, axis=-1, keep_dims=True)
    _logits = logits - _max_logits
    _logits = _logits * mask
    _exp_logits = tf.exp(_logits)

    exp_logits = _exp_logits * mask
    sum_exp_logits = tf.clip_by_value(
        tf.reduce_sum(exp_logits, axis=-1, keep_dims=True),
        1e-6,
        1e6)

    exp_gt_logits = _exp_logits * ground_truth
    sum_exp_gt_logits = tf.clip_by_value(
        tf.reduce_sum(exp_gt_logits, axis=-1, keep_dims=True),
        1e-6,
        1e6)

    _softmax = exp_logits / sum_exp_logits

    # _softmax and ground_truth are masked,
    # so need to mask the _logits
    _entropy = tf.reduce_sum(
        _softmax * (tf.log(sum_exp_logits) - _logits),
        axis=1,
        keep_dims=True)

    _loss = (tf.squeeze(tf.log(sum_exp_logits), axis=1)
            - tf.squeeze(tf.log(sum_exp_gt_logits), axis=1))

    # take care of case when all elements in mask is zero
    zeros1d = tf.fill(dims=[tf.shape(logits)[0]], value=0.0)
    one_prob = tf.concat(
        (tf.fill(dims=[tf.shape(logits)[0], 1],
                 value=1.0),
         tf.fill(dims=[tf.shape(logits)[0], tf.shape(logits)[1] - 1],
                 value=0.0)),
        axis=1)

    mask_sum = tf.reduce_sum(mask, axis=-1, keep_dims=False)

    softmax = tf.where(condition=tf.equal(mask_sum, zeros1d),
                       x=one_prob,
                       y=_softmax)
    entropy = tf.where(condition=tf.equal(mask_sum, zeros1d),
                       x=tf.expand_dims(zeros1d, axis=1),
                       y=_entropy)
    loss = tf.where(condition=tf.equal(mask_sum, zeros1d),
                    x=zeros1d,
                    y=_loss)

    if sum_weight_sample:
        assert weight_sample == [1.0, 1.0, 1.0]

        # making minimum 2 to avoid numerical instability
        # as log1 = 0 and weight is given by 1 / log(mask_sum)
        _mask_sum =  tf.clip_by_value(mask_sum, 2, 1e6)
        weight = 1 / tf.log(_mask_sum)
        weighted_loss = weight * loss
        cross_entropy_loss = tf.reduce_mean(weighted_loss)

    else:
        cross_entropy_loss = tf.reduce_mean(loss * weight_sample)

    return softmax, entropy, cross_entropy_loss


def get_feed_dict_rl(model,
                     obs,
                     is_training=False):
    """ Return the feed dict for rl algorithm

    Args:
        model: tensorflow graph made from PakcingPolicy
        obs (np.array[batch_size, obs_size]):
        is_training (bool): whether feed_dict for training or testing
    """

    # extracting required variable from the model
    rot_before_mov = model.rot_before_mov
    add_sum_fea = model.add_sum_fea
    fixed_fea_config = model.fixed_fea_config
    comp_pol_config = model.comp_pol_config

    # decoding the observation
    if sci.sparse.issparse(obs):
        obs = obs.toarray()

    obs_all = PackingEnv._decode_agent_obs(obs)

    # verifying if obsevations are compatible for the policy
    sha_step_typ = 1
    if rot_before_mov:
        rot_step_typ = 2
        mov_step_typ = 3
    else:
        rot_step_typ = 3
        mov_step_typ = 2

    possible_step_typ = []
    if comp_pol_config['sha_pol'] is not None:
        possible_step_typ.append(sha_step_typ)
    if comp_pol_config['mov_pol'] is not None:
        possible_step_typ.append(mov_step_typ)
    if comp_pol_config['rot_pol'] is not None:
        possible_step_typ.append(rot_step_typ)

    for i in range(obs_all['step_typ'].shape[0]):
        assert obs_all['step_typ'][i] in possible_step_typ

    if add_sum_fea:
        sha_vol = np.sum(obs_all["sha_rep"], axis=(2, 3, 4))
        cho_sha_vol = np.sum(obs_all["cho_sha_rep"], axis=(1, 2, 3))
        box_vol = np.sum(obs_all["box_rep"], axis=(1, 2, 3))

        total_vol = np.sum(sha_vol, axis=(1)) +  cho_sha_vol + box_vol
        sorted_sha_vol = np.sort(sha_vol, axis=1)
        sum_fea_vol = np.concatenate(
            (sorted_sha_vol,
             np.expand_dims(cho_sha_vol, 1),
             np.expand_dims(box_vol, 1)),
            axis=1
        )
        sum_fea = sum_fea_vol / np.expand_dims(total_vol, axis=1)

    if fixed_fea_config is not None:
        # extracting the fixed features
        batch_size = obs.shape[0]
        box_rep_size = fixed_fea_config['box_fea_dim']**3
        cho_sha_rep_size = ((fixed_fea_config['cho_sha_coarse_fea_dim']**3) +
                            (fixed_fea_config['cho_sha_fine_fea_dim']**3))

        new_box_rep = np.zeros((batch_size, box_rep_size))
        new_cho_sha_rep = np.zeros((batch_size, cho_sha_rep_size))
        new_sha_rep = np.zeros((batch_size, PackingEnv.MAX_NUM_SHA,
                                cho_sha_rep_size))

        for i in range(batch_size):
            new_box_rep[i] = packing_fea.extract_coarse_fea(
                voxel=obs_all["box_rep"][i],
                fea_per_dim=([fixed_fea_config["box_fea_dim"]] * 3))

            new_cho_sha_rep[i] = np.concatenate(
                (packing_fea.extract_coarse_fea(
                    voxel=obs_all["cho_sha_rep"][i],
                    fea_per_dim=(
                        [fixed_fea_config["cho_sha_coarse_fea_dim"]] * 3)),
                 packing_fea.extract_fine_fea(
                     voxel=obs_all["cho_sha_rep"][i],
                     fea_per_dim=(
                         [fixed_fea_config["cho_sha_fine_fea_dim"]] * 3))))

            for j in range(PackingEnv.MAX_NUM_SHA):
                if obs_all["sha_mask"][i, j] == 1:
                    new_sha_rep[i, j] = np.concatenate(
                        (packing_fea.extract_coarse_fea(
                            voxel=obs_all["sha_rep"][i, j],
                            fea_per_dim=(
                                [fixed_fea_config["cho_sha_coarse_fea_dim"]
                                 ] * 3)),
                         packing_fea.extract_fine_fea(
                             voxel=obs_all["sha_rep"][i, j],
                             fea_per_dim=(
                                 [fixed_fea_config["cho_sha_fine_fea_dim"]
                                  ] * 3))))

        obs_all["box_rep"] = new_box_rep
        obs_all["cho_sha_rep"] = new_cho_sha_rep
        obs_all["sha_rep"] = new_sha_rep

    feed_dict = {
        model.step_typ: obs_all["step_typ"],
        model.box_rep: obs_all["box_rep"],
        model.cho_rep: obs_all["cho_sha_rep"],
        model.sha_rep: obs_all["sha_rep"],
        model.sha_mask: obs_all["sha_mask"],
        model.pos_mov: obs_all["pos_mov"],
        model.pos_rot: obs_all["pos_rot"],
        model.is_training: is_training,
    }

    if add_sum_fea:
        feed_dict[model.sum_fea] = sum_fea

    return feed_dict


def sha_pol(
    box_enc_pol,
    box_enc_val,
    col_sha_enc_pol,
    col_sha_enc_val,
    sha_enc_pol,
    sha_mask,
    sum_fea,
    is_training,
    add_bn,
    NUM_FEA,
    add_sum_fea,
    reuse):
    """ Policy for selecting one shape out of the unchosen ones """

    with tf.variable_scope("sha_pol", reuse=reuse):
        _box_enc_pol = tf.expand_dims(box_enc_pol, axis=1)
        _box_enc_pol = tf.tile(_box_enc_pol,
                               [1, PackingEnv.MAX_NUM_SHA, 1])

        _col_sha_enc_pol = tf.expand_dims(col_sha_enc_pol, axis=1)
        _col_sha_enc_pol = tf.tile(_col_sha_enc_pol,
                                   [1, PackingEnv.MAX_NUM_SHA, 1])

        fea_act = tf.concat([_box_enc_pol,
                             _col_sha_enc_pol,
                             sha_enc_pol],
                            axis=2)

        if add_sum_fea:
            _sum_fea = tf.expand_dims(sum_fea, axis=1)
            _sum_fea = tf.tile(_sum_fea,
                               [1, PackingEnv.MAX_NUM_SHA, 1])

            fea_act = tf.concat([fea_act,
                                 _sum_fea],
                                axis=2)


        proc_fea_act = ser_fc_layers(
            inp=fea_act,
            is_training=is_training,
            num_hid=3,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=True,
            add_bn=add_bn,
            name="proc_fea_act",
            reuse=reuse)

        logits_act = final_fc_layers(
            inp=proc_fea_act,
            out_dim=1,
            name="logits_act",
            reuse=reuse)

        logits_act = tf.squeeze(logits_act, axis=2)

        logits_act = tf.concat((logits_act,
                                tf.fill(
                                    dims=[
                                        tf.shape(logits_act)[0],
                                        PackingEnv.NUM_MOV
                                        - PackingEnv.MAX_NUM_SHA
                                    ],
                                    value=0.0)),
                               axis=1)

        mask = tf.concat((sha_mask,
                          tf.fill(
                              dims=[
                                  tf.shape(logits_act)[0],
                                  PackingEnv.NUM_MOV
                                  - PackingEnv.MAX_NUM_SHA
                              ],
                              value=0.0)),
                         axis=1)

        fea_val = tf.concat([box_enc_val, col_sha_enc_val],
                            axis=1)

        if add_sum_fea:
            fea_val = tf.concat([fea_val, sum_fea],
                                axis=1)

        proc_fea_val = ser_fc_layers(
            inp=fea_val,
            is_training=is_training,
            num_hid=3,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=True,
            add_bn=add_bn,
            name="proc_fea_val",
            reuse=reuse)

        value = final_fc_layers(
            inp=proc_fea_val,
            out_dim=1,
            name="value",
            reuse=reuse)

    return logits_act, mask, value


def null_pol(box_enc_pol):

    logits_act = tf.fill(
        dims=[
            tf.shape(box_enc_pol)[0],
            PackingEnv.NUM_MOV
        ],
        value=0.0)

    mask = logits_act

    value = tf.fill(
        dims=[tf.shape(box_enc_pol)[0], 1],
        value=0.0)

    return logits_act, mask, value


def get_sample_type(step_typ, rot_before_mov):
    """ Return which samples are of which policy

    Args:
        step_typ (tensor[batch_size]):
        rot_before_mov (bool):
    """

    ones = tf.fill(
        dims=[tf.shape(step_typ)[0]],
        value=1.0)
    zeros = tf.fill(
        dims=[tf.shape(step_typ)[0]],
        value=0.0)

    sha_sample = tf.where(
        condition=tf.equal(step_typ, ones),
        x=ones,
        y=zeros)

    if rot_before_mov:
        rot_step_typ = 2
        mov_step_typ = 3
    else:
        rot_step_typ = 3
        mov_step_typ = 2

    mov_sample = tf.where(
        condition=tf.equal(step_typ, mov_step_typ * ones),
        x=ones,
        y=zeros)

    rot_sample = tf.where(
        condition=tf.equal(step_typ, rot_step_typ * ones),
        x=ones,
        y=zeros)

    return sha_sample, mov_sample, rot_sample


def comp_pol(
    step_typ,
    box_rep,
    cho_rep,
    sha_rep,
    sha_mask,
    pos_mov,
    pos_rot,
    sum_fea,
    ground_truth,
    is_training,
    NUM_FEA,
    rot_before_mov,
    add_bn,
    add_sum_fea,
    policy_weights,
    comp_pol_config,
    fixed_fea_config,
    reuse):

    # box encoding
    box_enc_pol = ser_fc_layers(
            inp=box_rep,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=2,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="box_enc_pol",
            reuse=reuse)
    box_enc_val = ser_fc_layers(
            inp=box_rep,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=2,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="box_enc_val",
            reuse=reuse)

    # chosen shape encoding
    cho_enc_pol = ser_fc_layers(
            inp=cho_rep,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=2,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="cho_enc_pol",
            reuse=reuse)
    cho_enc_val = ser_fc_layers(
            inp=cho_rep,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=2,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="cho_enc_val",
            reuse=reuse)

    # shape encoding
    sha_enc_pol = ser_fc_layers(
            inp=sha_rep,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=2,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="sha_enc_pol",
            reuse=reuse)
    sha_enc_val = ser_fc_layers(
            inp=sha_rep,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=2,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="sha_enc_val",
            reuse=reuse)

    col_sha_enc_pol = get_col_sha_enc(sha_enc_pol)
    col_sha_enc_val = get_col_sha_enc(sha_enc_val)

    # possible move encoding
    # Not actually used in the paper
    mov_enc_pol = encoder3d(
        pos_mov,
        is_training=is_training,
        add_bn=add_bn,
        feature_size=NUM_FEA,
        name="mov_enc_pol",
        reuse=reuse)
    mov_enc_val = encoder3d(
        pos_mov,
        is_training=is_training,
        add_bn=add_bn,
        feature_size=NUM_FEA,
        name="mov_enc_val",
        reuse=reuse)

    if rot_before_mov:
        rot_enc_pol = None
        rot_enc_val = None
    else:
        # possible rotation encoding
        rot_enc_pol = ser_fc_layers(
            inp=pos_rot,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=1,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="rot_enc_pol",
            reuse=reuse)
        rot_enc_val = ser_fc_layers(
            inp=pos_rot,
            is_training=is_training,
            add_bn=add_bn,
            num_hid=1,
            hid_dim=NUM_FEA,
            out_dim=NUM_FEA,
            relu_after=False,
            name="rot_enc_val",
            reuse=reuse)


    if comp_pol_config['sha_pol'] is None:
        sha_logits_act, sha_mask, sha_value = null_pol(box_enc_pol)

    else:
        sha_logits_act, sha_mask, sha_value = comp_pol_config['sha_pol'](
            box_enc_pol=box_enc_pol,
            box_enc_val=box_enc_val,
            col_sha_enc_pol=col_sha_enc_pol,
            col_sha_enc_val=col_sha_enc_val,
            sha_enc_pol=sha_enc_pol,
            sha_mask=sha_mask,
            sum_fea=sum_fea,
            is_training=is_training,
            add_bn=add_bn,
            NUM_FEA=NUM_FEA,
            add_sum_fea=add_sum_fea,
            reuse=reuse)

    if comp_pol_config['mov_pol'] is None:
        mov_logits_act, mov_mask, mov_value = null_pol(box_enc_pol)

    else:
        mov_logits_act, mov_mask, mov_value = comp_pol_config['mov_pol'](
            box_enc_pol=box_enc_pol,
            box_enc_val=box_enc_val,
            col_sha_enc_pol=col_sha_enc_pol,
            col_sha_enc_val=col_sha_enc_val,
            cho_enc_pol=cho_enc_pol,
            cho_enc_val=cho_enc_val,
            mov_enc_pol=mov_enc_pol,
            mov_enc_val=mov_enc_val,
            pos_mov=pos_mov,
            is_training=is_training,
            add_bn=add_bn,
            NUM_FEA=NUM_FEA,
            reuse=reuse)

    if comp_pol_config['rot_pol'] is None:
        rot_logits_act, rot_mask, rot_value = null_pol(box_enc_pol)

    else:
        rot_logits_act, rot_mask, rot_value = comp_pol_config['rot_pol'](
            box_enc_pol=box_enc_pol,
            box_enc_val=box_enc_val,
            col_sha_enc_pol=col_sha_enc_pol,
            col_sha_enc_val=col_sha_enc_val,
            cho_enc_pol=cho_enc_pol,
            cho_enc_val=cho_enc_val,
            rot_enc_pol=rot_enc_pol,
            rot_enc_val=rot_enc_val,
            pos_rot=pos_rot,
            sum_fea=sum_fea,
            is_training=is_training,
            add_bn=add_bn,
            NUM_FEA=NUM_FEA,
            rot_before_mov=rot_before_mov,
            add_sum_fea=add_sum_fea,
            reuse=reuse)

    sha_sample, mov_sample, rot_sample = get_sample_type(step_typ,
                                                         rot_before_mov)
    weight_sample = (policy_weights[0] * sha_sample) \
        + (policy_weights[1] * mov_sample) \
        + (policy_weights[2] * rot_sample)

    sha_sample = tf.expand_dims(sha_sample, 1)
    mov_sample = tf.expand_dims(mov_sample, 1)
    rot_sample = tf.expand_dims(rot_sample, 1)

    logits_act_all = (sha_logits_act * sha_sample) \
        + (mov_logits_act * mov_sample) \
        + (rot_logits_act * rot_sample)
    mask_all = (sha_mask * sha_sample) \
        + (mov_mask * mov_sample) \
        + (rot_mask * rot_sample)

    action_prob, entropy, cross_entropy_loss = masked_softmax_entropy_loss(
        logits=logits_act_all,
        mask=mask_all,
        ground_truth=ground_truth,
        weight_sample=weight_sample)

    _value = (sha_value * sha_sample) \
        + (mov_value * mov_sample) \
        + (rot_value * rot_sample)

    return action_prob, entropy, cross_entropy_loss, _value


def get_accuracy(ground_truth, action_best, step_typ, rot_before_mov):
    sha_sample, mov_sample, rot_sample = get_sample_type(step_typ,
                                                         rot_before_mov)

    _action_best = tf.one_hot(action_best, depth=(int)(PackingEnv.NUM_MOV))
    cor_pred = tf.reduce_sum(ground_truth * _action_best,
                             axis=1,
                             keep_dims=False)
    # cor_pred = tf.cast(tf.equal(ground_truth, action_best), tf.float32)
    sha_acc = tf.reduce_sum(cor_pred * sha_sample) / tf.reduce_sum(sha_sample)
    mov_acc = tf.reduce_sum(cor_pred * mov_sample) / tf.reduce_sum(mov_sample)
    rot_acc = tf.reduce_sum(cor_pred * rot_sample) / tf.reduce_sum(rot_sample)

    return sha_acc, mov_acc, rot_acc

# Source: "https://stackoverflow.com/questions/38559755/
#               how-to-get-current-available-gpus-in-tensorflow"
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# Source: "https://github.com/tensorflow/models/blob/master/
#               tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101"
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all
    towers. Note that this function provides a synchronization point across
    all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        ranges over the devices. The inner list ranges over the different
        variables.

    Returns:
            List of pairs of (gradient, variable) where the gradient has been
            averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]

# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign

class proba_distribution(object):
    pass


class PackingPolicy(object):
    """ A countom policy for the packing task """

    NUM_FEA = 128

    def __init__(self,
                 sess,
                 *args,
                 rot_before_mov=True,
                 add_bn=False,
                 add_sum_fea=True,
                 policy_weights=[1.0, 1.0, 1.0],
                 fixed_fea_config={
                    'box_fea_dim': 10,
                    'cho_sha_coarse_fea_dim': 8,
                    'cho_sha_fine_fea_dim': 8,
                 },
                 comp_pol_config={
                    'sha_pol': sha_pol,
                    'mov_pol': None,
                    'rot_pol': None
                 },
                 reuse=None,
                 **kwargs):
        """
            Args:
                policy_weights: list of weight [w1, w2, w3] to loss for each
                    policy, w1 is for sha, w2 is for mov and w3 is for rot
                fixed_fea_config: configuration for the fixed features.
                comp_pol_config: which function to use for which part. If
                    comp_pol_config['sha_pol'] is None than our policy will not
                    handle this
        """

        self.sess = sess
        self.rot_before_mov = rot_before_mov
        self.add_bn = add_bn
        self.add_sum_fea = add_sum_fea
        self.policy_weights = policy_weights
        self.fixed_fea_config = fixed_fea_config
        self.comp_pol_config = comp_pol_config
        self.reuse = reuse
        # assertions based on tested code
        assert fixed_fea_config is not None
        assert 'box_fea_dim' in fixed_fea_config
        assert 'cho_sha_coarse_fea_dim' in fixed_fea_config
        assert 'cho_sha_fine_fea_dim' in fixed_fea_config
        assert comp_pol_config['sha_pol'] is not None
        assert comp_pol_config['mov_pol'] is None
        assert comp_pol_config['rot_pol'] is None

        with tf.device('/cpu:0'):
            inputs = self.setup_placeholders()

        # additional placeholder
        self.is_training = tf.placeholder(tf.bool)

        available_gpus = get_available_gpus()

        # Source: https://github.com/vahidk/EffectiveTensorflow#multi_gpu
        input_splits = {}
        for k, v in inputs.items():
            input_splits[k] = tf.split(v, len(available_gpus), axis=0)

        action_prob_splits = []
        entropy_splits = []
        cross_entropy_loss_splits = []
        _value_splits = []

        # Source:
        # http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/
        for i, id in enumerate(available_gpus):
            _reuse = self.reuse or bool(i)
            with tf.variable_scope( "model", reuse=_reuse):
                # Source: "https://stackoverflow.com/questions/35919020/
                #   whats-the-difference-of-name-scope-and-a-variable
                #   -scope-in-tensorflow"
                # name scope is just a name similar to variable scope
                # However, name scope is ignored by tf.get_variable
                name = 'tower_{}'.format(i)
                # Use the assign_to_device function to ensure that variables
                # are created on the controller.
                with tf.device(assign_to_device(id, "/cpu:0")),  \
                    tf.name_scope(name) as scope:

                    action_prob, entropy, cross_entropy_loss, _value = \
                        comp_pol(
                            NUM_FEA=self.NUM_FEA,
                            rot_before_mov=self.rot_before_mov,
                            add_bn=self.add_bn,
                            add_sum_fea=self.add_sum_fea,
                            policy_weights=self.policy_weights,
                            comp_pol_config=self.comp_pol_config,
                            fixed_fea_config=self.fixed_fea_config,
                            reuse=self.reuse,
                            is_training=self.is_training,
                            **{k : v[i] for k, v in input_splits.items()})

                    # Retain the Batch Normalization updates operations only from the
                    # final tower. Ideally, we should grab the updates from all towers
                    # but these stats accumulate extremely fast so we can ignore the
                    # other stats from the other towers without significant detriment.
                    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION,
                                                          scope)

                    action_prob_splits.append(action_prob)
                    entropy_splits.append(entropy)
                    cross_entropy_loss_splits.append(cross_entropy_loss)
                    _value_splits.append(_value)

        self.batchnorm_updates_op = tf.group(*batchnorm_updates)
        # print("all batch norm update steps")
        # print(self.batchnorm_updates_op)

        with tf.device("/cpu:0"):
            self.action_prob = tf.concat(action_prob_splits, axis=0)
            self.entropy = tf.concat(entropy_splits, axis=0)
            self._value = tf.concat(_value_splits, axis=0)

            self.cross_entropy_loss = tf.stack(cross_entropy_loss_splits,
                                               axis=0)
            self.cross_entropy_loss = tf.reduce_mean(self.cross_entropy_loss)

            # sampling action
            action_dis = tf.distributions.Categorical(probs=self.action_prob)
            self.action = action_dis.sample()

            # getting the best action
            self._action_best = tf.argmax(self.action_prob, axis=1)

            # calculating negative log likelihood of action
            self.neglogp = self.get_neglogp(self.action)

            # things required for compatibility with ppo2
            self.pdtype = make_proba_dist_type(
                spaces.Discrete(PackingEnv.NUM_MOV))

            self.value_fn = self._value
            self._value = self.value_fn[:, 0]

            self.initial_state = None

            self.proba_distribution = proba_distribution()
            self.proba_distribution.neglogp = self.get_neglogp
            self.proba_distribution.entropy = self.get_entropy

            self.sha_acc, self.mov_acc, self.rot_acc = get_accuracy(
                self.ground_truth,
                self._action_best,
                self.step_typ,
                self.rot_before_mov)

    def setup_placeholders(self):
        """ For setting up the placeholders that are to be distributed across
            gpus

            Return:
                inputs (dict): associating each input placeholder with a name,
                    will be later used for multu-gpu support
        """

        self.step_typ = tf.placeholder(tf.float32, shape=[None])

        self.sha_mask = tf.placeholder(tf.float32,
                                       shape=[None,
                                              PackingEnv.MAX_NUM_SHA])
        self.pos_mov = tf.placeholder(tf.float32,
                                      shape=[None,
                                             PackingEnv.MOV_RES,
                                             PackingEnv.MOV_RES,
                                             PackingEnv.MOV_RES])
        self.pos_rot = tf.placeholder(tf.float32,
                                      shape=[None,
                                             PackingEnv.NUM_ROT])

        self.ground_truth = tf.placeholder(tf.float32,
                                      shape=[None,
                                            PackingEnv.NUM_MOV])
        # this palceholder although defined will not be used if add_sum_fea
        # is false
        self.sum_fea = tf.placeholder(tf.float32,
                              shape=[None,
                                     PackingEnv.MAX_NUM_SHA + 2])

        box_rep_size = self.fixed_fea_config['box_fea_dim'] ** 3
        cho_sha_rep_size = (
            (self.fixed_fea_config['cho_sha_coarse_fea_dim'] ** 3)
            + (self.fixed_fea_config['cho_sha_fine_fea_dim'] ** 3))

        self.box_rep = tf.placeholder(tf.float32,
                                      shape=[None,
                                             box_rep_size])
        self.cho_rep = tf.placeholder(tf.float32,
                                      shape=[None,
                                             cho_sha_rep_size])
        self.sha_rep = tf.placeholder(tf.float32,
                                      shape=[None,
                                             PackingEnv.MAX_NUM_SHA,
                                             cho_sha_rep_size])

        placeholders = {
            "step_typ": self.step_typ,
            "box_rep": self.box_rep,
            "cho_rep": self.cho_rep,
            "sha_rep": self.sha_rep,
            "sha_mask": self.sha_mask,
            "pos_mov": self.pos_mov,
            "pos_rot": self.pos_rot,
            "ground_truth": self.ground_truth,
            "sum_fea":self.sum_fea
        }

        return placeholders


    def step(self, obs, state=None, mask=None, deterministic=False):
        feed_dict = get_feed_dict_rl(
            model=self,
            obs=obs,
            is_training=False)

        action, value, neglogp = self.sess.run([self.action,
                                                self._value,
                                                self.neglogp],
                                               feed_dict)
        return action, value, None, neglogp


    def action_best(self, obs, state=None, mask=None, deterministic=False):
        feed_dict = get_feed_dict_rl(
            model=self,
            obs=obs,
            is_training=False)

        _action_best = self.sess.run(self._action_best,
                                    feed_dict)
        return _action_best


    def action_best_n(self, obs, n):
        num_obs = obs.shape[0]
        action = np.zeros((num_obs, n))
        score = np.zeros((num_obs, n))

        feed_dict = get_feed_dict_rl(
            model=self,
            obs=obs,
            is_training=False)
        # action_prob nparray(batch_size, NUM_MOV)
        action_prob = self.sess.run(self.action_prob, feed_dict)

        for i in range(num_obs):
            _action_prob = action_prob[i]
            _action = np.argsort(_action_prob)[::-1][0:n]
            _score = np.sort(_action_prob)[::-1][0:n]

            not_poss_act = (_score == 0)
            _action[not_poss_act] = -1
            _score[not_poss_act] = 1
            _score = np.log(_score)

            action[i] = _action
            score[i] = _score

        return action, score


    def action_all_sorted(self, obs):
        num_obs = obs.shape[0]
        feed_dict = get_feed_dict_rl(
            model=self,
            obs=obs,
            is_training=False)

        action_prob = self.sess.run(self.action_prob, feed_dict)
        action = []
        for i in range(num_obs):
            _action_prob = action_prob[i]
            num_poss_act = np.count_nonzero(_action_prob)
            _action = np.argsort(_action_prob)[::-1][0:num_poss_act]
            action.append(_action)

        return action


    def proba_step(self, obs, state=None, mask=None):
        feed_dict = get_feed_dict_rl(
            model=self,
            obs=obs,
            is_training=False)

        return self.sess.run(self.action_prob, feed_dict)


    def value(self, obs, state=None, mask=None):
        feed_dict = get_feed_dict_rl(
            model=self,
            obs=obs,
            is_training=False)

        return self.sess.run(self._value, feed_dict)


    def get_neglogp(self, action):
        """ Returns the negative log likelihood for an action

            Args:
                action_prob (tensor[batch_size, num_actions])
                action (tensor[batch_size]): each value from [0: num_aciton-1]
        """

        action_prob = self.action_prob

        action_one_hot = tf.one_hot(action,
                                    depth=int(action_prob.shape[-1]))

        neglogp = - tf.log(tf.reduce_sum(
            action_prob * tf.stop_gradient(action_one_hot),
            axis=1) + 1e-10)

        return neglogp


    def get_entropy(self):
        """ Returns entropy for each sample"""

        return self.entropy[:, 0]
