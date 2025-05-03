import tensorflow as tf

# 查看 TensorFlow 是否能检测到 GPU
print("TensorFlow 版本:", tf.__version__)
print("GPU 是否可用:", tf.config.list_physical_devices('GPU'))

# 显示详细的设备信息
print("\n可用的物理设备:")
for device in tf.config.list_physical_devices():
    print(device)

# 如果有 GPU，显示 GPU 详细信息
if tf.config.list_physical_devices('GPU'):
    # 显示 GPU 内存信息
    print("\nGPU 内存信息:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"设备: {gpu}")
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"设备详情: {gpu_details}")
        except:
            print("无法获取详细信息")
    
    # 测试一个简单的 GPU 操作
    print("\n执行 GPU 测试...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"GPU 计算结果: \n{c}")
else:
    print("\nTensorFlow 未检测到 GPU。")
    print("可能的原因:")
    print("1. 您的系统没有 GPU")
    print("2. GPU 驱动程序未正确安装")
    print("3. CUDA/cuDNN 与当前 TensorFlow 版本不兼容")
    print("4. 您安装的是 CPU 版本的 TensorFlow")