import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}

# weights_v = {
#     # Convolution Layers
#     'c1': tf.get_variable('W1_v', shape=(2, 2, 1, 4), \
#                           initializer=tf.contrib.layers.xavier_initializer()),
#     # Dense Layers
#     'd1': tf.get_variable('W2_v', shape=(2 * 2 * 4, 8),
#                           initializer=tf.contrib.layers.xavier_initializer()),
#     'out': tf.get_variable('W3_v', shape=(8, 4),
#             initializer=tf.contrib.layers.xavier_initializer())
#
# }
# weights_e = {
#     # Convolution Layers
#     'c1': tf.get_variable('W1_e', shape=(2, 2, 1, 4), \
#                           initializer=tf.contrib.layers.xavier_initializer()),
#     # Dense Layers
#     'd1': tf.get_variable('W2_e', shape=(2 * 2 * 4, 8),
#                           initializer=tf.contrib.layers.xavier_initializer()),
#     'out': tf.get_variable('W3_e', shape=(8, 4),
#             initializer=tf.contrib.layers.xavier_initializer())
#
# }
# biases_v = {
#     # Convolution Layers
#     'c1': tf.get_variable('B1_v', shape=(4), initializer=tf.zeros_initializer()),
#
#     # Dense Layers
#     'd1': tf.get_variable('B2_v', shape=(8), initializer=tf.zeros_initializer()),
#     'out': tf.get_variable('B3_v', shape=(4), initializer=tf.zeros_initializer()),
# }
#
# biases_e = {
#     # Convolution Layers
#     'c1': tf.get_variable('B1_e', shape=(4), initializer=tf.zeros_initializer()),
#
#     # Dense Layers
#     'd1': tf.get_variable('B2_e', shape=(8), initializer=tf.zeros_initializer()),
#     'out': tf.get_variable('B3_e', shape=(4), initializer=tf.zeros_initializer()),
# }


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv_net(data, weights, biases):
    # Convolution layers
    conv1 = conv2d(data, weights[0], biases[0])  # [4,4,4]

    # Flatten
    # flat = tf.reshape(conv1, [-1] + conv1.shape.as_list()[1:])
    flat = tf.reshape(conv1, [-1, weights[1].get_shape().as_list()[0]])
    # [2*2*4] = [16]

    # Fully connected layer
    fc1 = tf.add(tf.matmul(flat, weights[1]), biases[1])  # [8]
    fc1 = tf.nn.relu(fc1)  # [8]

    # Output
    out = tf.add(tf.matmul(fc1, weights[2]), biases[2]) # [4]
    return out


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Layer(object):
    def __init__(self, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.vars = []

    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs

    @abstractmethod
    def _call(self, inputs):
        pass


class Dense(Layer):
    def __init__(self, input_dim, output_dim, dropout=0.0, act=tf.nn.relu, name=None):
        super(Dense, self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        with tf.variable_scope(self.name):
            self.weight = tf.get_variable(name='weight', shape=(input_dim, output_dim), dtype=tf.float32)
            self.bias = tf.get_variable(name='bias', shape=output_dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight]

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        output = tf.matmul(x, self.weight) + self.bias
        return self.act(output)


class CrossCompressUnit(Layer):
    def __init__(self, dim, name=None):
        super(CrossCompressUnit, self).__init__(name)
        self.dim = dim
        self.filters_conv = 8 #28
        self.filters_dense = 16 #16
        with tf.variable_scope(self.name):
            self.weight_c1_v = tf.get_variable('W1_v', shape=(2, 2, 1, self.filters_conv), \
                              initializer=tf.contrib.layers.xavier_initializer())
            self.weight_d1_v = tf.get_variable('W2_v', shape=(self.dim/2 * self.dim/2 * self.filters_conv, self.filters_dense),
                              initializer=tf.contrib.layers.xavier_initializer())
            self.weight_out_v = tf.get_variable('W3_v', shape=(self.filters_dense, self.dim),
                               initializer=tf.contrib.layers.xavier_initializer())
            self.weight_c1_e = tf.get_variable('W1_e', shape=(2, 2, 1, self.filters_conv), \
                              initializer=tf.contrib.layers.xavier_initializer())
            self.weight_d1_e = tf.get_variable('W2_e', shape=(self.dim/2 * self.dim/2 * self.filters_conv, self.filters_dense),
                              initializer=tf.contrib.layers.xavier_initializer())
            self.weight_out_e = tf.get_variable('W3_e', shape=(self.filters_dense, self.dim),
                               initializer=tf.contrib.layers.xavier_initializer())
            self.biases_c1_v = tf.get_variable('B1_v', shape=(self.filters_conv), initializer=tf.zeros_initializer())
            self.biases_d1_v = tf.get_variable('B2_v', shape=(self.filters_dense ), initializer=tf.zeros_initializer())
            self.biases_out_v = tf.get_variable('B3_v', shape=(self.dim), initializer=tf.zeros_initializer())

            self.biases_c1_e = tf.get_variable('B1_e', shape=(self.filters_conv), initializer=tf.zeros_initializer())
            self.biases_d1_e = tf.get_variable('B2_e', shape=(self.filters_dense ), initializer=tf.zeros_initializer())
            self.biases_out_e = tf.get_variable('B3_e', shape=(self.dim), initializer=tf.zeros_initializer())

        self.vars = [self.weight_c1_v, self.weight_d1_v, self.weight_out_v, self.weight_c1_e, self.weight_d1_e, self.weight_out_e,
                     self.biases_c1_v, self.biases_d1_v, self.biases_out_v, self.biases_c1_e, self.biases_d1_e, self.biases_out_e]


        # with tf.variable_scope(self.name):
        #     self.weight_vv = tf.get_variable(name='weight_vv', shape=(dim, 1), dtype=tf.float32)
        #     self.weight_ev = tf.get_variable(name='weight_ev', shape=(dim, 1), dtype=tf.float32)
        #     self.weight_ve = tf.get_variable(name='weight_ve', shape=(dim, 1), dtype=tf.float32)
        #     self.weight_ee = tf.get_variable(name='weight_ee', shape=(dim, 1), dtype=tf.float32)
        #     self.bias_v = tf.get_variable(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
        #     self.bias_e = tf.get_variable(name='bias_e', shape=dim, initializer=tf.zeros_initializer())
        # self.vars = [self.weight_vv, self.weight_ev, self.weight_ve, self.weight_ee]

    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs
        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = tf.expand_dims(v, dim=2)
        e = tf.expand_dims(e, dim=1)
        # [batch_size, dim, dim]
        c_matrix = tf.matmul(v, e)
        c_matrix = tf.expand_dims(c_matrix, -1) # Our code

        # c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim, 1]
        # c_matrix = tf.reshape(c_matrix, [-1, self.dim])

        ## Our Implementation
        weights_v = [self.weight_c1_v, self.weight_d1_v, self.weight_out_v]
        weights_e = [self.weight_c1_e, self.weight_d1_e, self.weight_out_e]
        biases_v = [self.biases_c1_v, self.biases_d1_v, self.biases_out_v]
        biases_e = [self.biases_c1_e, self.biases_d1_e, self.biases_out_e]
        v_output = conv_net(c_matrix, weights_v, biases_v)
        e_output = conv_net(c_matrix, weights_e, biases_e)

        '''
        ### The paper's Code we want to change.
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])
        
        # [batch_size, dim]
        v_output = tf.reshape(tf.matmul(c_matrix, self.weight_vv) + tf.matmul(c_matrix_transpose, self.weight_ev),
                              [-1, self.dim]) + self.bias_v
        e_output = tf.reshape(tf.matmul(c_matrix, self.weight_ve) + tf.matmul(c_matrix_transpose, self.weight_ee),
                              [-1, self.dim]) + self.bias_e

        '''

        return v_output, e_output

