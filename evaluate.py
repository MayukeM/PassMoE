# 模型评估工具
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from config import Config, LEET_DICTIONARY
from model import PassMoEP
from data import PasswordDataset, load_password_data

class PasswordEvaluator:
    """密码生成模型评估器"""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.model_path = model_path or self.config.MODEL_SAVE_PATH
        
        # 加载模型
        self.model = self._load_model()
        
        # 加载评估数据
        _, self.val_passwords = load_password_data(self.config)
        self.sample_passwords = self.val_passwords[:1000]  # 使用1000个样本进行评估
        
        # 预生成一批密码用于评估
        self.generated_passwords = self._generate_evaluation_passwords()
        
    def _load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在")
            
        model = PassMoEP(
            base_model_name=self.config.BASE_MODEL_NAME,
            leet_dictionary=LEET_DICTIONARY,
            config=self.config
        )
        
        model.load_state_dict(torch.load(
            self.model_path, 
            map_location=self.config.DEVICE
        ))
        
        return model
    
    def _generate_evaluation_passwords(self, num_passwords=10000):
        """生成用于评估的密码集"""
        print(f"生成 {num_passwords} 个密码用于评估...")
        passwords = []
        
        # 生成无前缀密码
        passwords.extend(self.model.generate_passwords(num_passwords=num_passwords // 2))
        
        # 使用随机前缀生成
        prefixes = ["", "a", "1", "ab", "12", "abc", "123", "user", "pass", "admin"]
        for prefix in prefixes:
            passwords.extend(self.model.generate_passwords(
                prefix=prefix, 
                num_passwords=num_passwords // 20
            ))
        
        # 去重
        unique_passwords = list(set(passwords))
        print(f"去重后得到 {len(unique_passwords)} 个密码")
        
        return unique_passwords[:num_passwords]
    
    def calculate_coverage(self, top_n=10000):
        """计算模型生成密码覆盖测试集的比例"""
        # 取测试集的前top_n个密码
        test_set = set(self.sample_passwords[:top_n])
        # 取生成密码的前top_n个
        generated_set = set(self.generated_passwords[:top_n])
        
        # 计算交集
        overlap = test_set.intersection(generated_set)
        coverage = len(overlap) / len(test_set) if test_set else 0
        
        print(f"覆盖度@{top_n}: {coverage:.4f} ({len(overlap)}/{len(test_set)})")
        return coverage
    
    def evaluate_entropy_distribution(self):
        """评估生成密码的熵分布"""
        def calculate_entropy(password):
            if not password:
                return 0.0
            prob = [float(password.count(c)) / len(password) for c in set(password)]
            entropy = -sum(p * np.log2(p) for p in prob) if prob else 0.0
            return entropy
        
        # 计算生成密码的熵
        generated_entropies = [calculate_entropy(pwd) for pwd in self.generated_passwords]
        
        # 计算测试集密码的熵
        test_entropies = [calculate_entropy(pwd) for pwd in self.sample_passwords]
        
        # 绘制分布
        plt.figure(figsize=(10, 6))
        sns.histplot(generated_entropies, kde=True, label='生成密码', alpha=0.5)
        sns.histplot(test_entropies, kde=True, label='测试集密码', alpha=0.5)
        plt.xlabel('密码熵 (bits)')
        plt.ylabel('频率')
        plt.title('密码熵分布对比')
        plt.legend()
        plt.savefig('entropy_distribution.png')
        plt.close()
        
        print(f"生成密码平均熵: {np.mean(generated_entropies):.2f} bits")
        print(f"测试集密码平均熵: {np.mean(test_entropies):.2f} bits")
        
        return {
            'generated_mean': np.mean(generated_entropies),
            'test_mean': np.mean(test_entropies),
            'generated_std': np.std(generated_entropies),
            'test_std': np.std(test_entropies)
        }
    
    def evaluate_pattern_coverage(self):
        """评估生成密码对不同模式的覆盖能力"""
        # 定义密码模式检测器
        def detect_pattern(password):
            patterns = []
            
            # 检测数字
            if re.search(r'\d', password):
                patterns.append('数字')
                
            # 检测小写字母
            if re.search(r'[a-z]', password):
                patterns.append('小写字母')
                
            # 检测大写字母
            if re.search(r'[A-Z]', password):
                patterns.append('大写字母')
                
            # 检测特殊字符
            if re.search(r'[^a-zA-Z0-9]', password):
                patterns.append('特殊字符')
                
            # 检测PII特征
            pii_score = self.model._detect_pii(password)
            if pii_score > 0.5:
                patterns.append('PII特征')
                
            # 检测Leetspeak
            leet_score = self.model._detect_leetspeak(password)
            if leet_score > 0.3:
                patterns.append('Leetspeak')
                
            return patterns
        
        # 分析生成密码的模式
        generated_patterns = {}
        for pwd in self.generated_passwords:
            patterns = detect_pattern(pwd)
            for p in patterns:
                generated_patterns[p] = generated_patterns.get(p, 0) + 1
        
        # 分析测试集密码的模式
        test_patterns = {}
        for pwd in self.sample_passwords:
            patterns = detect_pattern(pwd)
            for p in patterns:
                test_patterns[p] = test_patterns.get(p, 0) + 1
        
        # 计算百分比
        generated_total = len(self.generated_passwords)
        generated_percent = {k: v/generated_total*100 for k, v in generated_patterns.items()}
        
        test_total = len(self.sample_passwords)
        test_percent = {k: v/test_total*100 for k, v in test_patterns.items()}
        
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        patterns = list(set(generated_percent.keys()).union(set(test_percent.keys())))
        
        x = np.arange(len(patterns))
        width = 0.35
        
        generated_vals = [generated_percent.get(p, 0) for p in patterns]
        test_vals = [test_percent.get(p, 0) for p in patterns]
        
        plt.bar(x - width/2, generated_vals, width, label='生成密码')
        plt.bar(x + width/2, test_vals, width, label='测试集密码')
        
        plt.xlabel('密码模式')
        plt.ylabel('包含该模式的密码百分比 (%)')
        plt.title('密码模式覆盖对比')
        plt.xticks(x, patterns, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('pattern_coverage.png')
        plt.close()
        
        return {
            'generated': generated_percent,
            'test': test_percent
        }
    
    def run_full_evaluation(self):
        """运行完整评估流程"""
        print("开始完整评估...")
        
        # 计算覆盖度
        coverage_1k = self.calculate_coverage(top_n=1000)
        coverage_10k = self.calculate_coverage(top_n=10000)
        
        # 评估熵分布
        entropy_stats = self.evaluate_entropy_distribution()
        
        # 评估模式覆盖
        pattern_stats = self.evaluate_pattern_coverage()
        
        # 保存评估结果
        results = {
            'coverage@1k': coverage_1k,
            'coverage@10k': coverage_10k,
            'entropy': entropy_stats,
            'patterns': pattern_stats
        }
        
        # 保存为JSON
        import json
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("评估完成，结果已保存至 evaluation_results.json")
        print("生成了以下可视化文件:")
        print("- entropy_distribution.png: 密码熵分布对比图")
        print("- pattern_coverage.png: 密码模式覆盖对比图")
        
        return results

if __name__ == "__main__":
    evaluator = PasswordEvaluator()
    evaluator.run_full_evaluation()
    