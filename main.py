# 主程序入口
from config import Config, LEET_DICTIONARY
from data import load_password_data, create_data_loaders
from model import PassMoEP
from trainer import Trainer
import torch
import os

def main():
    # 初始化配置
    config = Config()
    print(f"使用设备: {config.DEVICE}")
    print(f"基础模型: {config.BASE_MODEL_NAME}")
    
    # 加载数据
    print("加载密码数据...")
    train_passwords, val_passwords = load_password_data(config)
    
    # 创建模型
    print("初始化PassMoE-P模型...")
    model = PassMoEP(
        base_model_name=config.BASE_MODEL_NAME,
        leet_dictionary=LEET_DICTIONARY,
        config=config
    )
    
    # 如果存在预训练模型，加载它
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"加载预训练模型: {config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(
            config.MODEL_SAVE_PATH, 
            map_location=config.DEVICE
        ))
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        train_passwords, 
        val_passwords, 
        model.tokenizer, 
        config
    )
    
    # 初始化训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, config)
    trained_model = trainer.train()
    
    # 测试密码生成
    print("\n生成示例密码:")
    generated = trained_model.generate_passwords(prefix="", num_passwords=10)
    for i, pwd in enumerate(generated, 1):
        print(f"{i}. {pwd}")
    
    # 测试带有前缀的密码生成
    print("\n生成带有前缀的密码:")
    prefixes = ["Zhang", "P@ss", "a7b"]
    for prefix in prefixes:
        generated = trained_model.generate_passwords(prefix=prefix, num_passwords=3)
        print(f"前缀 '{prefix}':")
        for i, pwd in enumerate(generated, 1):
            print(f"  {i}. {pwd}")

if __name__ == "__main__":
    main()
    