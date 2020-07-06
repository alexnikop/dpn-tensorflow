
""" Tensorflow implementation of Dual Path Networks
Based on original MXNet implementation https://github.com/cypw/DPNs
"""

from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, \
    Activation, Dropout, Conv2D, Add, Input, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import sys

def get_model_params(model_type):

    small = False

    if model_type == 'dpn68':
        init_filters = 10
        G = 32
        depths = [3, 4, 12, 3]
        k = [16, 32, 32, 64]
        filters = [128, 128, 64]
        small = True

    elif model_type == 'dpn92':
        init_filters = 64
        G = 32
        depths = [3, 4, 20, 3]
        k = [16, 32, 24, 128]
        filters = [96, 96, 256]

    elif model_type == 'dpn98':
        init_filters = 96
        G = 40
        depths = [3, 6, 20, 3]
        k = [16, 32, 32, 128]
        filters = [160, 160, 256]

    elif model_type == 'dpn107':
        init_filters = 128
        G = 50
        depths = [4, 8, 20, 3]
        k = [20, 64, 64, 128]
        filters = [200, 200, 256]

    elif model_type == 'dpn131':
        init_filters = 128
        G = 40
        depths = [4, 8, 28, 3]
        k = [16, 32, 32, 128]
        filters = [160, 160, 256]
    
    else:
        print('model type must be in [dpn68, dpn92, dpn98, dpn107, dpn132].. exiting..')
        sys.exit(1)

    return init_filters, G, depths, k, filters, small

def bn_activation(bottom, name, bn_axis=-1, activation='relu'):
    bn = BatchNormalization(axis=bn_axis, name=name)(bottom)
    
    if activation is not None:
        return Activation(activation)(bn)
    else:
        return bn
    
def conv_operation(bottom, filters, ksize, strides, name, padding='same', use_bias=False):

    x = Conv2D(
        filters=filters, 
        kernel_size=(ksize, ksize),
        strides=(strides, strides),
        name=name,
        padding=padding,
        use_bias=use_bias)(bottom)

    return x

def group_conv(bottom, filters, ksize, strides, G, name, iter_num, padding='same'):
    total_conv = []
    filters_per_path = filters // G
    bn = bn_activation(bottom, name='{}_bn'.format(name))

    if iter_num == 0 and strides == 2:
        bn = tf.pad(bn, [[0,0], [1,1], [1,1], [0,0]])
        padding = 'valid'
    
    for i in range(G):
        input_split = bn[:, :, :, i * filters_per_path : (i + 1) * filters_per_path]
        conv =  conv_operation(input_split, filters_per_path, ksize, strides, '{}_{}'.format(name, i+1), padding=padding)
        total_conv.append(conv)
    
    final_conv = tf.concat(total_conv, axis=3)
    return final_conv

def dpn_block(bottom, filters, strides, G, k, blocks, scope, padding='same'):
    conv_id = int(scope.split('_')[-1]) + 1
    dense_layers = []
    dpn = bottom
    bn = bn_activation(bottom, name='conv{}_proj_bn'.format(conv_id))
    project = conv_operation(bn, filters[2] + 2 * k, 1, strides, 'conv{}_proj'.format(conv_id), padding)
    
    shortcut = project[:, :, :, :filters[2]]
    dense_layers.append(project[:, :, :, filters[2]:])

    for i in range(blocks):
        dpn = bn_activation(dpn, name='conv{}_{}_{}_bn'.format(conv_id, i+1, 1))
        dpn = conv_operation(dpn, filters[0], 1, 1, 'conv{}_{}_{}'.format(conv_id, i+1, 1))
        dpn = group_conv(dpn, filters[1], 3,
                    strides if i == 0 else 1, G,
                    'group_conv{}_{}'.format(conv_id, i+1), i)

        dpn = bn_activation(dpn, name='conv{}_{}_{}_bn'.format(conv_id, i+1, 2))
        dpn = conv_operation(dpn, filters[2] + k,  1, 1, 'conv{}_{}_{}'.format(conv_id, i+1, 2))

        residual = dpn[:, :, :, :filters[2]]
        dense = dpn[:, :, :, filters[2]:]
        residual = residual + shortcut
        shortcut = residual
        dense_layers.append(dense)
        dpn = tf.concat([residual] + dense_layers, axis=-1)
    
    return dpn


def dpn_model(input_shape=(224, 224, 3), model_type='dpn92', include_top=True, num_classes=1000):

    drop_rate = 0.5

    init_filters, G, depths, k,\
        filters, small = get_model_params(model_type)

    if small:
        init_filt_size = 3
    else:
        init_filt_size = 7


    inputs = Input(input_shape)
    x = conv_operation(inputs, init_filters, init_filt_size, 2, 'conv_input', 'same')
    x = bn_activation(x, name='bn_input')

    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool_input', padding='same')(x)

    for i in range(len(depths)):
        strides = 1 if i == 0 or i == 3 else 2
        x = dpn_block(x, filters, strides, G, k[i], depths[i], 'dpn_block_{}'.format(i+1))
        filters = [2 * x for x in filters]

    x = bn_activation(x, name='final_bn')
    
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dropout(drop_rate)(x)
        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
        x = conv_operation(x, num_classes, 1, 1, name='classifier', padding='same', use_bias=True)
        x = tf.squeeze(x, axis=(1, 2))
        x = Activation('softmax')(x)

    model = Model(inputs, x, name='dpn')

    return model
