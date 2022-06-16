from keras import layers
from keras.layers import *
from keras.models import *
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

IMAGE_ORDERING = 'channels_last'


def identity_block(input_tensor, filter_num, block):
    # 残差网络结构
    conv_name_base = 'res' + block + '_branch'
    in_name_base = 'in' + block + '_branch'

    # 1x1压缩
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(input_tensor)
    x = Conv2D(filter_num, (3, 3), data_format=IMAGE_ORDERING, name=conv_name_base + '2a')(x)
    x = InstanceNormalization(axis=3, name=in_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(filter_num, (3, 3), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = InstanceNormalization(axis=3, name=in_name_base + '2c')(x)

    # 残差网络
    x = layers.add([x, input_tensor])

    x = Activation('relu')(x)
    return x


def get_resnet(input_height, input_width, channel):
    # 生成网络构建
    img_input = Input(shape=(input_height, input_width, 3))

    # 提取特征
    # 128,128,3 -> 128,128,64
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 下采样
    # 128,128,64 -> 64,64,128 通道数改变 步长为2
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING, strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 64,64,128 -> 32,32,256
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    for i in range(9):
        x = identity_block(x, 256, block=str(i))

    # 将上面提取的特征进行利用和复原
    # 上采样
    # 32,32,256 -> 64,64,128
    x = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(x)
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 64,64,128 -> 128,128,64
    x = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(x)
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 128,128,64 -> 128,128,3
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(channel, (7, 7), data_format=IMAGE_ORDERING)(x)
    x = Activation('tanh')(x)
    model = Model(img_input, x)
    return model
