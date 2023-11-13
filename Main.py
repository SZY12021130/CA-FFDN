#-*- coding : utf-8-*-
import math
from encodings import unicode_escape
coding:unicode_escape
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
from keras.models import Model
from keras.layers import Reshape, Dense, multiply, Permute, Lambda, GlobalMaxPooling3D
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Concatenate,
    GlobalAveragePooling3D)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)
from keras import backend as K, layers


# （1）通道注意力
def channel_attenstion(inputs, ratio=4):
    '''ratio代表第一个全连接层下降通道数的倍数'''
    #print(inputs)
    init = inputs
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis].value
    #channel = inputs.shape[-2]  # 获取输入特征图的通道数,也就是filters个数

    # 分别对输出特征图进行全局最大池化和全局平均池化
    # [h,w,c]==>[None,c]
    x_max = GlobalMaxPooling3D()(inputs)
    x_avg = GlobalAveragePooling3D()(inputs)

    # [None,c]==>[1,1,c]
    c_shape = (1,1,1,filters)
    x_max = Reshape(c_shape)(x_max)  # -1代表自动寻找通道维度的大小
    x_avg = Reshape(c_shape)(x_avg)  # 也可以用变量channel代替-1
    x_max = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x_max)
    x_avg = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x_avg)
    x_max = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x_max)
    x_avg = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x_avg)

    # 第一个全连接层通道数下降1/4, [1,1,c]==>[1,1,c//4]
    #x_max = Dense(filters // ratio)(x_max)
    #x_avg = Dense(filters // ratio)(x_avg)

    # relu激活函数
    #x_max = Activation('relu')(x_max)
    #x_avg = Activation('relu')(x_avg)

    # 第二个全连接层上升通道数, [1,1,c//4]==>[1,1,c]
    #x_max = Dense(filters)(x_max)
    #x_avg = Dense(filters)(x_avg)

    # 结果在相叠加 [1,1,c]+[1,1,c]==>[1,1,c]
    x = layers.Add()([x_max, x_avg])##这是keras的函数
    #x =x_max + x_avg##这是keras的函数

    # 经过sigmoid归一化权重
    x = Lambda(lambda x:layers.activations.sigmoid(x))(x)
    #print(K.is_keras_tensor(x))
    # 输入特征图和权重向量相乘，给每个通道赋予权重
    x = layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]
    #x = inputs * x
    return x


# （2）空间注意力机制
def spatial_attention(inputs):
    # 在通道维度上做最大池化和平均池化[b,h,w,c]==>[b,h,w,1]
    # keepdims=Fale那么[b,h,w,c]==>[b,h,w]
    x_max = Lambda(lambda x:K.max(x, axis=-1, keepdims=True))(inputs)# 在通道维度求最大值
    x_avg = Lambda(lambda x:K.mean(x, axis=-1, keepdims=True))(inputs) # axis也可以为-1
    #print(K.is_keras_tensor(x_avg))
    #print(K.is_keras_tensor(x_max))
    # 在通道维度上堆叠[b,h,w,2]
    x = layers.concatenate([x_max, x_avg])
   # print(K.is_keras_tensor(x))
    # 1*1卷积调整通道[b,h,w,1]
    x = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=1, padding='same')(x)
    #print(K.is_keras_tensor(x))
    # sigmoid函数权重归一化
    x = Lambda(lambda x:layers.activations.sigmoid(x))(x)
    #print(K.is_keras_tensor(x))
    # 输入特征图和权重相乘
    x = layers.Multiply()([inputs, x])
    #print(K.is_keras_tensor(x))

    return x


# （3）CBAM注意力
def CBAM_attention(inputs):
    # 先经过通道注意力再经过空间注意力
    x = channel_attenstion(inputs)
    x = spatial_attention(x)
    return x
#SE注意力机制
def squeeze_excite_block(input, ratio=8):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    # print("1" * 10, filters)
    # 160

    # 光谱和空间采样
    se_shape = (1, 1, 1, filters)

    # 空间采样
    # se_shape = (1, 1, filters)

    # Squeeze
    se = GlobalAveragePooling3D()(init)
    # print(se.shape)
    # (?, 96)
    se = Reshape(se_shape)(se)

    # Excitation
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    # print(se.shape)
    # (?, 1, 1, 96)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    # 分配权重
    # print(se)
    # Tensor("dense_2/Sigmoid:0", shape=(?, 1, 1, 1, 160), dtype=float32)
    # print(init)
    # Tensor("conv1_block1_concat/concat:0", shape=(?, 5, 5, 99, 96), dtype = float32)

    x = multiply([init, se])
    # print(x.shape)
    # (?, 5, 5, 99, 96)

    return x

#ECA注意力机制
def eca_block(inputs, b=1, gama=2):
    # 输入特征图的通道数
    in_channel = inputs.shape[-1].value

    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size

    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1

    # [None,h,w,d,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling3D()(inputs)

    # [None,c]==>[None,c,1]
    x = layers.Reshape(target_shape=(in_channel, 1 ))(x)

    # [None,c,1]==>[None,c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    # sigmoid激活# [None,c,1]==>[None,c,1]
    x = Lambda(lambda x:layers.activations.sigmoid(x))(x)

    # [None,c,1]==>[None,1,1,1,c]
    x = layers.Reshape((1, 1, 1, in_channel))(x)

    # 结果和输入相乘[None,1,1,1,c]==>[None,h,w,d,c]
    outputs = layers.multiply([inputs, x])
    #outputs =

    return outputs
#改进的注意力特征融合
def eca_block1(inputs,inputs2, b=1, gama=2):
    # 输入特征图的通道数
    in_channel = inputs.shape[-1].value

    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size

    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1

    # [None,h,w,d,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling3D()(inputs)

    # [None,c]==>[None,c,1]
    x = layers.Reshape(target_shape=(in_channel, 1 ))(x)

    # [None,c,1]==>[None,c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    # sigmoid激活# [None,c,1]==>[None,c,1]
    x = Lambda(lambda x:layers.activations.sigmoid(x))(x)

    # [None,c,1]==>[None,1,1,1,c]
    x = layers.Reshape((1, 1, 1, in_channel))(x)

    # 结果和输入相乘[None,1,1,1,c]==>[None,h,w,d,c]
    outputs = layers.multiply([inputs2, x])
    #outputs =

    return outputs

def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4

#transition里的卷积
def _convbnrelu(x, nb_filters, stride, kernel_size, name):
    """
    Convolution block of the first layer

    :param x: input tensor
    :param nb_filters: integer or tuple, number of filters
    :param stride: integer or tuple, stride of convolution
    :param kernel_size: integer or tuple, filter's kernel size
    :param name: string, block label

    :return: output tensor of a block
    """
    x = Conv3D(filters=nb_filters, strides=stride, kernel_size=kernel_size, padding='same',
               kernel_initializer='he_normal', use_bias=False, name=name + "_conv3d", )(x)
    x = BatchNormalization(name=name + "_batch_norm")(x)
    x = Activation(activation='relu', name=name + '_relu')(x)
    return x


def _bottleneck(x, growth_rate, stride, name):
    """
    DenseNet-like block for subsequent layers
    :param x: input tensor
    :param growth_rate: integer, number of output channels
    :param stride: integer, stride of 3x3 convolution
    :param name: string, block label

    :return: output tensor of a block
    """
    x = Conv3D(filters=4 * growth_rate, strides=1, kernel_size=1, padding='same',
               kernel_initializer='he_normal', use_bias=False, name=name + "_conv3d_1x1x1")(x)
    x = BatchNormalization(name=name + "_batch_norm_1")(x)
    x = Activation(activation='relu', name=name + "_relu_1")(x)
    '''
    x = Conv3D(filters=growth_rate, strides=stride, kernel_size=(3, 3, 3),
               padding='same', kernel_initializer='he_normal', use_bias=False,
               name=name + "_conv3d_3x3x3")(x)
    x = BatchNormalization(name=name + "_batch_norm_2")(x)
    x = Activation(activation='relu', name=name + "_relu_2")(x)
    '''
    #这里尝试采用深度可分离

    x = Conv3D(filters=1, strides=stride, kernel_size=(3,3,1),
               padding='same', kernel_initializer='he_normal', use_bias=False,
               name=name + "_conv3d_3x3x1")(x)
    x = BatchNormalization(name=name + "_batch_norm_2")(x)
    x = Activation(activation='relu', name=name + "_relu_2")(x)

    x = Conv3D(filters=growth_rate, strides=1, kernel_size=(1, 1, 3),
              padding='same', kernel_initializer='he_normal', use_bias=False,
              name=name + "_conv3d_1x1x3")(x)
    x = BatchNormalization(name=name + "_batch_norm_3")(x)
    x = Activation(activation='relu', name=name + "_relu_3")(x)

    return x


def  basic_block(x, l_growth_rate=None, scale=2, name="basic_block"):
    """
    Basic building block of MSDNet

    :param x: Input tensor or list of tensors
    :param l_growth_rate: list, numbers of output channels for each scale
    :param scale: Number of different scales features
    :param name:
    :return: list of different scales features listed from fine-grained to coarse
    """
    output_features = []

    try:
        is_tensor = K.is_keras_tensor(x)
        # check if not a tensor
        # if keras/tf class raise error instead of assign False
        if not is_tensor:
            raise TypeError("Tensor or list [] expected")

    except ValueError:
        # if not keras/tf class set False
        is_tensor = False

    if is_tensor:#返回的x为含有三个tensor的组合列表，不是tensor了
        #生成n=1的垂直 三组 特征图
        for i in range(scale):
            mult = 2 ** i #2的i次方
            x = _convbnrelu(x, nb_filters=32 * mult, stride=min(2, mult), kernel_size=3, name=name + "_" + str(i))
            output_features.append(x)

    else:

        assert len(l_growth_rate) == scale, "Must be equal: len(l_growth_rate)={0} scale={1}".format(len(l_growth_rate),
                                                                                                     scale)
        #每次生成三组特征图
        for i in range(scale):
            if i == 0:
                #conv是stride为1，是横向特征图
                conv = _bottleneck(x[i], growth_rate=l_growth_rate[i], stride=1,
                                   name=name + "_conv3d_" + str(i))
                bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
                #采用的是瓶颈密集连接，卷积之后要concat一下
                conc = Concatenate(axis=bn_axis, name=name + "_concat_post_" + str(i))([conv, x[i]])
                #output_features.append(conc)
                #conc = eca_block(conc)
            else:
                #stride_conv是向斜下方的stride为2的卷积，是斜着生成特征图
                strided_conv = _bottleneck(x[i - 1], growth_rate=l_growth_rate[i], stride=2,
                                           name=name + "_strided_conv3d_1" + str(i))

                #conv是scale为2的时候，横向生成的特征图
                conv = _bottleneck(x[i], growth_rate=l_growth_rate[i], stride=1,
                                   name=name + "_conv3d_" + str(i))

                #注意力特征融合
                #aff = layers.add([strided_conv, conv])
                '''
                aff = layers.add([strided_conv, conv])
                sig = eca_block1(aff)#sigmoid
                aff1 = layers.multiply([sig, strided_conv])
                aff2 = layers.multiply([(1-sig), conv])
                aff3 = layers.add([aff1, aff2])
                '''


                #aff = layers.add([strided_conv, conv])#aff = strided_conv + conv
                #aff = eca_block(aff) #融合后的特征经过注意力机制
                 #融合后的特征经过注意力机制
                #aff1 = layers.multiply([aff, strided_conv]) #aff1 = aff * strided_conv
                #aff2 = layers.multiply([aff, conv])#aff2 = aff * conv
                #aff3 = layers.add([aff1, aff2])#aff3 = aff1 + aff2


                #conv2是垂直下采样
                #conv2 = _bottleneck(conc, growth_rate=l_growth_rate[i], stride=2,
                                           #name=name + "_strided_conv3d_2" + str(i))

                strided_conv = eca_block(strided_conv)
                conv = eca_block(conv)
                aff = layers.concatenate([strided_conv, conv ])

                bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
                #conc是横着的瓶颈结构先concat，再加上斜着的下采样卷积结果
                conc = Concatenate(axis=bn_axis, name=name + "_concat_pre_" + str(i))([aff,x[i]])
                #conc = Concatenate(axis=bn_axis, name=name + "_concat_pre_" + str(i))([strided_conv,conv, x[i]])
                #conc = eca_block(conc)
            output_features.append(conc)

    return output_features


def transition_block(x, reduction, name):
    """
    Transition block for network reduction
    :param x: list, set of tensors
    :param reduction: float, fraction of output channels with respect to number of input channels
    :param name: string, block label

    :return: list of tensors
    """
    output_features = []
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    for i, item in enumerate(x):

        conv = _convbnrelu(item, nb_filters=int(reduction * K.int_shape(item)[bn_axis]), stride=1, kernel_size=1,
                           name=name + "_transition_block_" + str(i))
        output_features.append(conv)

    return output_features


def classifier_block(x, nb_filters, nb_classes, activation, name):
    """
    Classifier block
    :param x: input tensor
    :param nb_filters: integer, number of filters
    :param nb_classes: integer, number of classes
    :param activation: string, activation function
    :param name: string, block label

    :return: block tensor
    """
    #两次深度可分离，添加注意力机制
    x = _convbnrelu(x, nb_filters=nb_filters, stride=2, kernel_size=3, name=name + "_1")
    #squeeze
    #x = _convbnrelu(x, nb_filters=16, stride=2, kernel_size=1, name=name + "_1")
    #expand
    #x = _convbnrelu(x, nb_filters=80, stride=1, kernel_size=1, name=name + "_2")
    #x = _convbnrelu(x, nb_filters=80, stride=1, kernel_size=3, name=name + "_3")
    x = _convbnrelu(x, nb_filters=nb_filters, stride=2, kernel_size=3, name=name + "_2")
    #x = _convbnrelu(x, nb_filters=16, stride=2, kernel_size=1, name=name + "_4")
    # expand
    #x = _convbnrelu(x, nb_filters=80, stride=1, kernel_size=1, name=name + "_5")
    #x = _convbnrelu(x, nb_filters=80, stride=1, kernel_size=3, name=name + "_6")


    #x = channel_attenstion(x)
    #x = _convbnrelu(x, nb_filters=1, stride=2, kernel_size=(3, 3, 1), name=name + "_3")
    # x = spatial_attention(x)
    #x = _convbnrelu(x, nb_filters6=nb_filters, stride=2, kernel_size=(1, 1, 3), name=name + "_4")
    # x = channel_attenstion(x)
    ###############################

    x = AveragePooling3D(pool_size=2, strides=2, padding='same', name=name + '_avg_pool3d')(x)
    x = Flatten(name=name + "_flatten")(x)
    out = Dense(units=nb_classes, activation=activation, name=name + "_dense")(x)
    return out


# 组合模型
class ResnetBuilder(object):
    @staticmethod
    def build(input_size, nb_classes=16, scale=2, depth=4, l_growth_rate=(6, 12,),
              transition_block_location=(3,4,5,6), classifier_ch_nb=128, classifier_location=(4,)):
        """
        Function that builds MSDNet

        :param input_size: tuple of integers, 3x1, size of input image
        :param nb_classes: integer, number of classes
        :param scale: integer, number of network's scales
        :param depth: integer, network depth
        :param l_growth_rate: tuple of integers, scale x 1, growth rate of each scale
        :param transition_block_location: tuple of integer, array of block's numbers to place transition block after
        :param classifier_ch_nb: integer, output channel of conv blocks in classifier, if None than the same number as in
                                          an input tensor
        :param classifier_location: tuple of integers, array of block's numbers to place classifier after

        :return: MSDNet
        """
        print('original input shape:', input_size)
        _handle_dim_ordering()
        if len(input_size) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_size)

        if K.image_dim_ordering() == 'tf':
            input_size = (input_size[1], input_size[2], input_size[3], input_size[0])
        print('change input shape:', input_size)

        inp = Input(shape=input_size)
        out = []

        for i in range(depth):

            if i == 0:
                #进入到循环（循环3次）
                #i==0的时候返回的x，是n=1的特征图组合
                x = basic_block(inp, l_growth_rate=[],
                                scale=scale, name="basic_block_" + str(i + 1))
            elif i in transition_block_location:
                x = transition_block(x, reduction=0.5, name="transition_block_" + str(i + 1))

                x = basic_block(x, l_growth_rate=l_growth_rate,
                                scale=scale, name="basic_block_" + str(i + 1))
                #scale -= 1
                #l_growth_rate = l_growth_rate[1:]
                #x = x[1:]
            else:
                if i in (1,):
                    #生成n=2的三组特征图之前的concat特征图，返回的x是一个列表
                    x = basic_block(x, l_growth_rate=l_growth_rate,
                                    scale=scale, name="basic_block_" + str(i + 1))
                    #进行n=2的三组特征图，降低通道数、返回的x是一个列表
                    x = transition_block(x, reduction=0.5, name="transition_block_" + str(i + 1))
                elif i == 2:
                    #返回的X是一个列表
                    x = basic_block(x, l_growth_rate=l_growth_rate,
                                    scale=scale, name="basic_block_" + str(i + 1))

            if i + 1 in classifier_location:
                bn_axis = 4 if K.image_data_format() == 'channels_last' else 1

                cls_ch = K.int_shape(x[-1])[bn_axis] if classifier_ch_nb is None else classifier_ch_nb
                final = _bottleneck(x[0],growth_rate=l_growth_rate[-1], stride=2,name='final')
                #final_ = Concatenate(axis=bn_axis, name= "final_" )([final, x[0]])
                final_connect = Concatenate(axis=bn_axis, name= 'final_connect') ([final, x[-1]])
                #x = x.append(final_connect)
                #final_connect = squeeze_excite_block(final_connect)
                #final_connect = eca_block(final_connect)
                #final_connect = channel_attenstion(final_connect)
                out.append(classifier_block(final_connect, nb_filters=cls_ch, nb_classes=nb_classes, activation='sigmoid',
                                            name='classifier_' + str(i + 1)))

        return Model(inputs=inp, outputs=out)

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs)


def main():
    model = ResnetBuilder.build_resnet_8((1, 13, 13, 30), 16)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary(positions=[.60, .88, 1, 1.])


if __name__ == '__main__':
    main()
