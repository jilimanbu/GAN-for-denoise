# 含噪图像路径
noise_data_path = "D:thesis//dataset//image//noise"  
# 真实图像路径
real_data_path = "D:thesis//dataset//image//real" 
# 图像高度
IMAGE_H = 256
# 图像宽度
IMAGE_W = 384
# 批处理量大小
BATCH_SIZE = 32 
# patch高度
PATCH_H = 64
# patch宽度
PATCH_W = 64
# patch轴向范围
H_AXIS= int(IMAGE_H/PATCH_H)
W_AXIS = int(IMAGE_W/PATCH_W)
# 图片格式
image_shape = (256, 384, 3)
# 输入通道数
input_nc = 3
# 输出通道数
output_nc = 3
# 生成器输入格式
input_shape_generator = (64, 64, input_nc)
# 判别器输入格式
input_shape_discriminator = (256, 384, output_nc)
# 残差网络个数
n_blocks_gen = 3
# 生成器滤波器个数（特征图个数）
ngf = 64
# 判别器滤波器个数
ndf = 64
# 迭代周期
EPOCHS = 2
# 峰值信噪比系数
fp=1
# 对抗损失系数
fa=0.1
