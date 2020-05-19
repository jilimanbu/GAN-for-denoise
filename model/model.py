```python

from keras.models import Model
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, Activation,Conv2DTranspose, BatchNormalization, Add, Subtract, UpSampling2D, Dropout,Cropping2D,Lambda,Reshape,Concatenate
from keras.layers.merge import Add
from keras.utils import conv_utils
from keras.losses import mean_squared_error
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.preprocessing import image
from keras.optimizers import Adam,RMSprop


def res_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    """残差网络块，二层卷积结构"""
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
    x = Activation('relu')(x)
    merged = Add()([inputs, x])
    return merged


def generator_model():
    """生成网络"""
    # 输入层,输入一张图像
    inputs = Input(shape=input_shape_generator)
            
    # 输入卷积层
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # 卷积层
    x = Conv2D(filters=ngf, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=ngf*2, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # 三层残差层
    for i in range(n_blocks_gen): 
        x = res_block(x, ngf*2, use_dropout=True)
    # 反卷积层
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(filters=ngf*2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(filters=ngf, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # 输出层
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='same')(x)
    x = Activation('tanh')(x)
    
    
    # 残差连接  
    outputs = Add()([inputs, x])
    outputs = Lambda(lambda z: z/2)(outputs)
    
    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model

def discriminator_model():
    """判别网络"""
    # 多输入接受生成图像以及真实图像
    inputs = Input(shape=(64,64,3), name='d_input_g')
    
    x = Conv2D(filters=ndf, kernel_size=(3, 3), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=ndf, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=ndf*2, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(filters=ndf*4, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)


    x = Conv2D(filters=ndf*8, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding='same')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)def GAN_model(generator, discriminator):
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    
    return model 
    

def GAN_model(generator, discriminator):
    """生成对抗网络"""
    input_noise = Input(shape=(64,64,3), name='gan_input_g')
    generated_image = generator(input_noise)
    outputs = discriminator(generated_image)
    model = Model(inputs=input_noise, outputs=[generated_image,outputs], name='GAN')
    return model

```
