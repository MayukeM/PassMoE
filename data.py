# 数据处理模块
import re
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import random

class PasswordDataset(Dataset):
    """密码数据集类，负责加载数据和特征提取"""
    
    def __init__(self, passwords, tokenizer, max_length=32):
        self.passwords = passwords
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 添加自定义特殊token
        self.tokenizer.add_special_tokens({
            "pad_token": "<PAD>", 
            "eos_token": "<EOS>"
        })
        
    def __len__(self):
        return len(self.passwords)
    
    def __getitem__(self, idx):
        password = self.passwords[idx] + "<EOS>"  # 添加结束符
        encoding = self.tokenizer(
            password,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 提取特征用于门控网络
        features = self._extract_features(password)
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten(),
            "features": torch.tensor(features, dtype=torch.float32)
        }
    
    def _extract_features(self, password):
        """提取密码的语义特征：PII得分、Leetspeak得分、结构熵"""
        pii_score = self._detect_pii(password)
        leet_score = self._detect_leetspeak(password)
        entropy = self._calculate_entropy(password)
        return [pii_score, leet_score, entropy]
    
    def _detect_pii(self, password):
        """检测密码中是否包含可能的PII（姓名、生日等）"""
        year_patterns = [r'\b19\d{2}\b', r'\b20\d{2}\b']
        name_fragments = ['john', 'mary', 'zhang', 'li', 'wang']
        
        score = 0.0
        for pattern in year_patterns:
            if re.search(pattern, password):
                score += 0.3
                
        for fragment in name_fragments:
            if fragment in password.lower():
                score += 0.3
                
        return min(score, 1.0)
    
    def _detect_leetspeak(self, password):
        """检测密码中是否包含Leetspeak转换"""
        from config import LEET_DICTIONARY
        leet_chars = set()
        for mappings in LEET_DICTIONARY.values():
            leet_chars.update(mappings)
            
        count = 0
        for char in password:
            if char in leet_chars:
                count += 1
                
        return min(count / max(len(password), 1), 1.0)
    
    def _calculate_entropy(self, password):
        """计算密码的香农熵"""
        if not password:
            return 0.0
        
        # 计算每个字符的概率
        prob = [float(password.count(c)) / len(password) for c in set(password)]
        # 计算香农熵
        entropy = -sum(p * math.log2(p) for p in prob)
        
        # 归一化到0-1范围
        return min(entropy / 8.0, 1.0)  # 假设最大熵为8 bits per character


def load_password_data(config):
    """加载密码数据并创建数据加载器"""
    # 设置随机种子
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    
    # 检查数据文件是否存在，不存在则创建示例数据
    if not os.path.exists(config.DATA_PATH):
        print(f"数据文件 {config.DATA_PATH} 不存在，创建示例数据...")
        create_example_data(config.DATA_PATH)
    
    # 加载数据
    df = pd.read_csv(config.DATA_PATH)
    passwords = df["password"].dropna().tolist()
    print(f"加载了 {len(passwords)} 个密码样本")
    
    # 划分训练集和验证集
    train_passwords, val_passwords = train_test_split(
        passwords, 
        test_size=config.TEST_SIZE, 
        random_state=config.SEED
    )
    
    return train_passwords, val_passwords

def create_example_data(file_path):
    """创建示例密码数据（从外部文件读取）"""
    pd.DataFrame({"password": [line.strip() for line in open(os.path.join(os.path.dirname(__file__), "data", "password_templates.txt"), 'r') if line.strip()]}).to_csv(file_path, index=False)



def create_data_loaders(train_passwords, val_passwords, tokenizer, config):
    """创建训练和验证数据加载器"""
    train_dataset = PasswordDataset(
        train_passwords, 
        tokenizer, 
        max_length=config.MAX_LENGTH
    )
    val_dataset = PasswordDataset(
        val_passwords, 
        tokenizer, 
        max_length=config.MAX_LENGTH
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader
    