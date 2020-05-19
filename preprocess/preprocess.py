import random
import matplotlib.pyplot as plt
import os
import math
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import keras.backend as K


def image_get(images_path, train_rate=0.8, test_rate=0.2):
    """ 根据路径读取图片，返回RBG图像的图片张量，变型为[IMAGE_SIZE, IMAGE_SIZE]，归一化到[-1,1] """
    # 读取全部含噪训练图像路径名称列表
    images_name_list = glob.glob(os.path.join(images_path, "*"))
    file_length = len(images_name_list)
    train_list = []
    for i in range(int(file_length*train_rate)):
        image_path = images_name_list[i]
        image_format_L = Image.open(image_path)
        if image_format_L.mode != 'L':
            image_format_L.conver('L')
        image_format_RGB = image_format_L.convert('RGB')
        image = np.array(image_format_RGB)
        image = (image - 127.5) / 127.5
        train_list.append(image)
    test_list = []
    for i in range(file_length-int(file_length*test_rate),file_length):
        image_path = images_name_list[i]
        image_format_L = Image.open(image_path)
        if image_format_L.mode != 'L':
            image_format_L.conver('L')
        image_format_RGB = image_format_L.convert('RGB')
        image = np.array(image_format_RGB)
        image = (image - 127.5) / 127.5
        test_list.append(image)
    return np.array(train_list).astype(np.float16), np.array(test_list).astype(np.float16)
    

def array2img(image_array):
    """由数组列表变为图片对象"""
    image = image_array*127.5+127.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image, mode='RGB')
    image = image.convert('L')
    return image
    
    
def rgb2l(image_array):
    image = image_array*127.5+127.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image, mode='RGB')
    image = image.convert('L')
    array = np.array(image)
    return array
    
    
def patch_make_overlap(image_array, h_axis=H_AXIS, w_axis=W_AXIS,patch_h=PATCH_H,patch_w=PATCH_W):
    """将一张图提取为重合的patch"""
    patch_list = []
    for i in np.arange(1,h_axis-1.5,0.5):
        for j in np.arange(1,w_axis-1.5,0.5):
            patch = image_array[int(i*patch_h):int((i+1)*patch_h),int(j*patch_w):int((j+1)*+patch_w)]
            patch_list.append(patch)
    return np.array(patch_list)


def patch_make(image_array, patch_height=64, patch_weight=64):
    """将一张图分割为不重合的patch"""
    patch_list = []
    height_axis = int(image_array.shape[0]/patch_height)
    weight_axis = int(image_array.shape[1]/patch_weight)
    for i in range(height_axis):
        for j in range(weight_axis):
            patch = image_array[i*patch_height:(i+1)*patch_height,j*patch_weight:(j+1)*+patch_weight]
            patch_list.append(patch)
    return np.array(patch_list)


def patch_dataset_make(image_list):
    length = len(image_list)-1
    patch_dataset = patch_make_overlap(image_list[0])
    for i in range(length):
        patch_list = patch_make_overlap(image_list[i+1])
        patch_dataset = np.concatenate([patch_dataset, patch_list])
    return patch_dataset
    
   
def rebuilt(patch_list,patch_height=64,patch_weight=64):
    """由一个图片块数组列表还原为图片"""
    height_axis = int(IMAGE_H/patch_height)
    weight_axis = int(IMAGE_W/patch_weight)
    row_rebuilt = patch_list[0]
    for j in range(weight_axis-1):
        row_rebuilt = np.hstack((row_rebuilt,patch_list[j+1]))
    image_rebuilt = np.array(row_rebuilt)
    for i in range(height_axis-1):
        row_rebuilt = patch_list[(i+1)*6]
        for j in range(weight_axis-1):
            row_rebuilt = np.hstack((row_rebuilt, patch_list[6*(i+1)+j+1]))
        image_rebuilt = np.vstack((image_rebuilt,row_rebuilt))
    image_rebuilt = array2img(image_rebuilt)
    return image_rebuilt


def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    d_loss=plt.subplot(2,2,1)
    g_loss=plt.subplot(2,2,2)
    d_loss.plot(d_losses, label='Discriminitive loss')
    g_loss.plot(gan_losses[0], label='GAN loss')
    plt.legend()
    
    
def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)
    
   
def plotGeneratedImages(epoch,batch_size=4):
    for i in range(2,3):
        noise = noise_train_patch_list[batch_size*i:batch_size*i+1]
        generate_image = g.predict(noise)
        generate_image = array2img(generate_image[0])
        noise_image = array2img(noise_train_patch_list[batch_size*i])
        real_image = array2img(real_train_patch_list[batch_size*i])
        noise_all = array2img(noise_train_data[i])
        real_all = array2img(real_train_data[i])
        plt.figure()
        gen = plt.subplot(1,3,1)
        noi = plt.subplot(1,3,2)
        rea = plt.subplot(1,3,3)
        gen.imshow(generate_image, cmap='gray')
        noi.imshow(noise_image, cmap='gray')
        rea.imshow(real_image, cmap='gray')
        plt.show()
        plt.imshow(noise_all, cmap='gray')
        plt.show()
        plt.imshow(real_all, cmap='gray')
        plt.show()
        g_in = patch_make(noise_train_data[2])
        gene = g.predict(g_in)
        gene = rebuilt(gene)
        plt.imshow(gene, cmap='gray')
        plt.show()
        plt.tight_layout()
        
        
