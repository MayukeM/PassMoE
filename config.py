# 配置参数
import torch

class Config:
    # 模型配置
    BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # 基础模型
    LORA_RANK = 32  # LoRA的秩
    MAX_LENGTH = 32  # 密码最大长度
    HIDDEN_DIM = 1024  # 专家网络隐藏层维度
    
    # 训练配置
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 5e-4
    GATING_LR_SCALE = 2.0  # 门控网络学习率缩放因子
    LEET_LOSS_LAMBDA = 0.1  # Leetspeak损失权重
    GRADIENT_CLIP = 1.0  # 梯度裁剪阈值
    
    # 生成配置
    TEMPERATURE = 1.0  # 生成温度
    TAU = 0.01  # 概率阈值
    NUM_CANDIDATES = 5  # 每个步骤的候选token数量
    
    # 数据配置
    DATA_PATH = "passwords.csv"
    TEST_SIZE = 0.2  # 验证集比例
    SEED = 42  # 随机种子
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存配置
    MODEL_SAVE_PATH = "best_passmoe_p.pt"
    LOG_PATH = "training_logs.csv"

# Leetspeak转换字典
LEET_DICTIONARY = {
    'a': ['a', '@', '4'],
    'b': ['b', '8'],
    'c': ['c', '(', '{', '['],
    'e': ['e', '3'],
    'g': ['g', '6', '9'],
    'i': ['i', '1', '!'],
    'l': ['l', '1', '|'],
    'o': ['o', '0'],
    's': ['s', '5', '$'],
    't': ['t', '7', '+'],
    'z': ['z', '2']
}
    