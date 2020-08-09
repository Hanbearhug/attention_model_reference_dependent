import datetime
import numpy as np
import tensorflow as tf
from tqdm import notebook
"""
日期:20200809
作者：韩方园
备注：weight权重明显是peak较大，但却出现了end权重较小，而first权重较大的现象，与逻辑不符，需要进一步观察
"""


def get_past_window_feature(r, raw_data, lag_days, genre_type):
    """
    获取过去一段时间内的评分列表
    :param r:
    :param raw_data:
    :param lag_days:
    :param genre_type: 两种，total: 特定用户过去一段时间所有评分， same: 特定用户过去一段时间相同类型电影评分
    :return:
    """
    begin_date = r['date'] - datetime.timedelta(lag_days)
    end_date = r['date']

    if genre_type=='total':
        df_fea = raw_data.loc[(raw_data['userId'] == r['userId']) & (raw_data['date'] < end_date) &
                              (raw_data['date'] >= begin_date)]
    else:
        df_fea = raw_data.loc[(raw_data['userId'] == r['userId']) & (raw_data['date'] < end_date) &
                              (raw_data['date'] >= begin_date) & (raw_data[r['genres_list']].sum(axis=1) > 0)]
    if len(df_fea) ==0:
        return []
    else:
        return df_fea['rating'].tolist()

def get_past_movie_rating(r, raw_data, lag_days):
    """
    获得特定电影过去一段时间的评分均值
    :param r:
    :param raw_data:
    :param lag_days:
    :return:
    """
    begin_date = r['date'] - datetime.timedelta(lag_days)
    end_date = r['date']

    df_fea = raw_data.loc[(raw_data['movieId'] == r['movieId']) & (raw_data['date'] < end_date) &
                          (raw_data['date'] >= begin_date)]
    if len(df_fea) == 0:
        return None
    else:
        return df_fea['rating'].mean()

def get_agg_feature(r, feature_type, k=None):
    if len(r) == 0:
        return None
    r_list = [float(num) for num in r.split(',')]
    if feature_type == 'peak':
        return np.max(r_list)
    elif feature_type == 'end':
        return r_list[0]
    elif feature_type == 'avg':
        return np.mean(r_list)
    elif feature_type == 'sum':
        return np.sum(r_list)
    elif feature_type == 'top_k_avg':
        if len(r_list) <= k:
            return np.mean(r_list)
        else:
            return np.mean(np.sort(r_list)[-k:])
    elif feature_type == 'end_k_avg':
        if len(r_list) <= k:
            return np.mean(r_list)
        else:
            return np.mean(r_list[:k])
    else:
        raise ValueError("不存在的特征类型")

def past_score_feature(r, raw_data, genre_type, feature_type, lag_day, total_genres):
    """
    分各个类型的电影评价统计peak, sum, avg, end特征，如果genre为total，则不区分类型, 否则统计与当前电影具有相同特征的电影评分,
    此函数可以弃用，效率太低
    genre: 电影类型是否为同类型
    feature_type: peak、sum、avg、end
    lag_day: 过去的时间窗口，时间窗口为[begin_date, current_date) 注: current_date --> r['date']
    """
    begin_date = r['date'] - datetime.timedelta(lag_day)
    current_date = r['date']
    genres_list = [genre for genre in r['genres_list'] if genre in total_genres]
    """
    结合时间窗口和特征类型得到用于计算特征的表格
    """
    if feature_type == 'end':
        df_fea = raw_data.loc[(raw_data['userId']==r['userId']) & (raw_data['date']<current_date)]
    elif genre_type == 'total':
        df_fea = raw_data.loc[(raw_data['userId']==r['userId']) & (raw_data['date']<current_date) & (raw_data['date']>=begin_date)]
    elif genre_type == 'same':
        df_fea = raw_data.loc[(raw_data['userId']==r['userId']) & (raw_data['date']<current_date) &
                              (raw_data['date']>=begin_date) & (raw_data[genres_list].sum(axis=1)>0)]
    else:
        raise ValueError("没有这种类型的genre选择方式")
    """
    分特征类型计算
    """
    if len(df_fea)==0:
            return None
    else:
        if feature_type == 'end':
            df_fea.sort_values(['date'], ascending=False, inplace=True)
            df_fea.reset_index(drop=True, inplace=True)
            return df_fea.loc[0, 'rating']
        elif feature_type == 'peak':
            return df_fea['rating'].max()
        elif feature_type == 'avg':
            return df_fea['rating'].mean()
        elif feature_type == 'sum':
            return df_fea['rating'].sum()
        else:
            raise ValueError("不存在的特征类型")


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, embedding_size):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(embedding_size)[np.newaxis, :],
                            embedding_size)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, mask):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # mask shape = (batch_size, max_len)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        score += tf.expand_dims(mask, 2) * (-1e9)

        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class ReferAttention:
    def __init__(self, base_feature_nums, att_feature_nums1, att_feature_nums2, num_class, learn_rate=0.01,
                 use_base=True,
                 att_units=4, hidden_size1=30, is_position=False):
        self.base_feature_nums = base_feature_nums
        self.att_feature_nums1 = att_feature_nums1
        self.att_feature_nums2 = att_feature_nums2
        self.num_class = num_class
        self.learn_rate = learn_rate
        self.use_base = use_base
        self.att_units = att_units
        self.hidden_size1 = hidden_size1
        self.is_position = is_position
        self._init_graph()

    def _init_graph(self):
        tf.reset_default_graph()
        self.x_base = tf.placeholder(tf.float32, [None, self.base_feature_nums], name='x_base')
        self.key_input1 = tf.placeholder(tf.float32, [None, None, self.att_feature_nums1], name='att_input1')
        self.key_input2 = tf.placeholder(tf.float32, [None, None, self.att_feature_nums2], name='att_input2')
        self.y_true = tf.placeholder(tf.int32, [None, ], name='y_true')
        self.query_input1 = tf.placeholder(tf.float32, [None, self.att_feature_nums1], name='query_input1')
        self.query_input2 = tf.placeholder(tf.float32, [None, self.att_feature_nums2], name='query_input2')

        if self.use_base:
            self.key_max1 = tf.reduce_max(self.key_input1, axis=1)  # [batch_size, att_feature_nums1]
            self.key_end1 = self.key_input1[:, 0, :]  # [batch_size, att_feature_nums1]

            self.key_max2 = tf.reduce_max(self.key_input2, axis=1)  # [batch_size, att_feature_nums2]
            self.key_end2 = self.key_input2[:, 0, :]  # [batch_size, att_feature_nums2]

            self.l1 = tf.concat(
                [
                    self.x_base,
                    tf.subtract(self.query_input1, self.key_max1),
                    tf.subtract(self.query_input1, self.key_end1),
                    tf.subtract(self.query_input2, self.key_max2),
                    tf.subtract(self.query_input2, self.key_end2)
                ],
                axis=1,
                name='l1'
            )
        else:
            self.key_att1 = self._attention(self.query_input1, self.key_input1, self.key_input1, self.att_units)
            self.key_att2 = self._attention(self.query_input2, self.key_input2, self.key_input2, self.att_units)
            self.l1 = tf.concat(
                [
                    self.x_base,
                    tf.subtract(self.query_input1, self.key_att1),
                    tf.subtract(self.query_input2, self.key_att2)
                ],
                axis=1,
                name='l1'
            )

        self.l2 = tf.layers.dense(
            self.l1,
            self.hidden_size1,
            activation=tf.nn.relu,
            name='l2'
        )

        self.output = tf.layers.dense(
            self.l2,
            self.num_class,
            name='output'
        )
        self.y_pred = tf.nn.softmax(self.output, name='y_pred')
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output,
                labels=self.y_true,
            ),
            name='loss'
        )
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        init = tf.initializers.global_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        return

    def _attention(self, queries, keys, values, num_units):
        """
        :param queries:
        :param keys:
        :param values:
        :param num_units:
        :return: [B, P]
        """
        Q = tf.layers.dense(tf.expand_dims(queries, axis=1), num_units, activation=tf.nn.relu)  #
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  #
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu)  #

        weights = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        weights = weights / (K.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))

        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(weights) * (-2 ** 32 + 1)
        weights = tf.where(tf.equal(key_masks, 0), paddings, weights)
        self.weights = tf.nn.softmax(weights)
        outputs = tf.matmul(self.weights, V)
        return tf.reshape(outputs, [-1, num_units])

    def _get_batch(self, data, tar_col, base_cols, series_cols_list, len_list, batch_size=1024, q=30, is_training=True):
        step = int(len(data) / batch_size) + 1
        # index = np.random.permutation(len(data))
        # data = data.iloc[index].reset_index(drop=True)
        for i in range(step):
            begin = i * batch_size
            end = min((i + 1) * batch_size, len(data))

            x_base = data.loc[begin:end - 1, base_cols].values.astype(np.float32)
            if is_training:
                y_base = data.loc[begin:end - 1, tar_col].values.astype(np.int32)

            x_list = []
            mask_list = []
            q_list = []
            for j in range(len(series_cols_list)):
                series_cols = series_cols_list[j]
                len_col = len_list[j]
                max_len = int(np.percentile(data.loc[begin:end - 1, len_col], q))
                if max_len < 2:
                    max_len = 2
                series = np.array([])
                mask_series = []
                for m in range(begin, end):
                    len_m = data.loc[m, len_col]
                    if len_m == 0:
                        series_m = np.zeros((1, max_len, len(series_cols)))  # [1, max_len, depth]
                        mask_m = [1] * (max_len - 1)
                    elif len_m < max_len:
                        series_m = np.array([[float(r) for r in data.loc[m, col].split(',')] for col in
                                             series_cols]).T  # [len_m, depth]
                        series_m = np.concatenate(
                            [series_m, np.zeros((max_len - len_m, len(series_cols)))])  # [max_len, depth]
                        series_m = np.expand_dims(series_m, axis=0)  # [1, max_len, depth]
                        mask_m = [0] * (len_m - 1) + [1] * (max_len - len_m)
                    else:
                        series_m = np.array([[float(r) for r in data.loc[m, col].split(',')] for col in
                                             series_cols]).T  # [len_m, depth]
                        series_m = series_m[:max_len, :]
                        series_m = np.expand_dims(series_m, axis=0)  # [1, max_len, depth]
                        mask_m = [0] * (max_len - 1)
                    try:
                        series = np.concatenate([series, series_m], axis=0)  # [batch_size, max_len, depth]
                    except:
                        series = series_m
                    mask_series.append(mask_m)

                x_series = series[:, 1:, :] / 5  # [batch_size, max_len-1, depth]
                q_series = series[:, 0, :]  # [batch_size, 1, depth]
                mask_series = np.array(mask_series)  # [batch_size, max_len-1]

                if self.is_position:
                    depth = len(series_cols)
                    mask_series_tile = np.repeat(np.expand_dims(mask_series, axis=2), repeats=depth,
                                                 axis=2)  # [batch_size, max_len-1, depth]
                    position_enc = np.array([
                        [pos / np.power(10000, 2. * i / depth) for i in range(depth)]
                        for pos in range(max_len - 1)])  # [max_len-1, depth]
                    position_enc[:, 0::2] = np.float32(np.sin(position_enc[:, 0::2]))  # dim 2i
                    position_enc[:, 1::2] = np.float32(np.cos(position_enc[:, 1::2]))  # dim 2i+1
                    position_enc = np.repeat(np.expand_dims(position_enc, axis=0), repeats=end - begin,
                                             axis=0)  # [batch_size, max_len-1, depth]
                    position_pad = np.zeros_like(position_enc)
                    self.position_enc = np.where(mask_series_tile == 1, position_pad, position_enc)
                    x_series += self.position_enc

                x_list.append(x_series.astype(np.float32))
                mask_list.append(mask_series.astype(np.float32))
                q_list.append(q_series.astype(np.float32) / 5)
            if is_training:
                yield y_base, x_base, x_list, q_list, mask_list
            else:
                yield x_base, x_list, q_list, mask_list

    def fit_on_batch(self, x_base_batch, x_list_batch, q_list_batch, mask_list_batch=None, y_batch=None):
        batch_loss, _ = self.sess.run([self.loss, self.train_op],
                                      feed_dict={self.x_base: x_base_batch,
                                                 self.key_input1: x_list_batch[0],
                                                 self.key_input2: x_list_batch[1],
                                                 self.query_input1: q_list_batch[0],
                                                 self.query_input2: q_list_batch[1],
                                                 self.y_true: y_batch})
        return batch_loss

    def fit(self, data, tar_col, base_cols, series_cols_list, len_list, batch_size=1024, q=30, epoches=20):
        steps = int(len(data) / batch_size) + 1
        total_losses = []

        for epoch in range(epoches):
            total_loss = []
            perm = np.random.permutation(np.arange(len(data)))
            data_perm = data.iloc[perm].reset_index(drop=True)

            for (batch, (y_batch, x_base_batch, x_list_batch, q_list_batch, mask_list_batch)) in notebook.tqdm(
                    enumerate(self._get_batch(
                        data_perm, tar_col, base_cols, series_cols_list, len_list, batch_size=batch_size, q=q
                    )), total=steps):
                batch_loss = self.fit_on_batch(x_base_batch, x_list_batch, q_list_batch, y_batch=y_batch)
                total_loss.append(batch_loss)

                if batch % 20 == 0:
                    y_pred = self.sess.run(
                        self.y_pred,
                        feed_dict={self.x_base: x_base_batch,
                                   self.key_input1: x_list_batch[0],
                                   self.key_input2: x_list_batch[1],
                                   self.query_input1: q_list_batch[0],
                                   self.query_input2: q_list_batch[1]}
                    )
                    y_true_sparse = np.zeros((batch_size, self.num_class))
                    for i in range(batch_size):
                        y_true_sparse[i, y_batch[i]] = 1
                    y_pred_label = np.argmax(y_pred, axis=1)

                    print(
                        f'epoch {epoch} batch {batch} loss: {np.mean(total_loss)}  auc: {round(roc_auc_score(y_true_sparse, y_pred, average="weighted"), 4)} acc{round(np.mean(y_batch == y_pred_label), 4)}')
            if epoch % 5 == 0:
                print(f'epoch {epoch} loss: {np.mean(total_loss)}')
            total_losses.append(np.mean(total_loss))
        return total_losses

    def predict(self, test_data, base_cols, series_cols_list, len_list, q=90):
        y_pred = np.array([])
        for x_base_batch, x_list_batch, q_list_batch, mask_list_batch in self._get_batch(test_data, tar_col, base_cols,
                                                                                         series_cols_list, len_list, q,
                                                                                         is_training=False):
            y_pred_batch = self.sess.run(
                self.y_pred,
                feed_dict={self.x_base: x_base_batch,
                           self.key_input1: x_list_batch[0],
                           self.key_input2: x_list_batch[1],
                           self.query_input1: q_list_batch[0],
                           self.query_input2: q_list_batch[1]}
            )
            try:
                y_pred = np.vstack((y_pred, y_pred_batch))
            except:
                y_pred = y_pred_batch
        return y_pred

    def save(self, export_dir):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, export_dir)
        print(f"Model saved in path {export_dir}")
        return

    def load(self, import_dir):
        saver = tf.train.Saver()
        saver.restore(self.sess, import_dir)
        return
