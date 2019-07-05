# constantInit.py


INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点
LAYER1_NODE = 500    # 隐藏层数



# 模型相关的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
#TRAINING_STEPS = 5000

MOVING_AVERAGE_DECAY = 0.99


num_classes = 10
img_rows, img_cols = 28, 28

BATCH_SIZE = 128     # 每次batch打包的样本个数
TRAINING_STEPS = 20