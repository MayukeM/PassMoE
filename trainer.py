# 训练模块
import torch
import torch.optim as optim
import pandas as pd
import os
from tqdm import tqdm
import csv
from datetime import datetime

class Trainer:
    """模型训练器类，负责模型的训练和验证"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 初始化日志
        self._init_logging()
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self):
        """创建优化器"""
        return optim.AdamW([
            {"params": self.model.experts.parameters(), "lr": self.config.LEARNING_RATE},
            {"params": self.model.gating_network.parameters(), 
             "lr": self.config.LEARNING_RATE * self.config.GATING_LR_SCALE}
        ])
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.EPOCHS
        )
    
    def _init_logging(self):
        """初始化训练日志"""
        if not os.path.exists(self.config.LOG_PATH):
            with open(self.config.LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'train_loss', 'val_loss', 'lr'])
    
    def _log_metrics(self, epoch, train_loss, val_loss):
        """记录训练指标"""
        lr = self.optimizer.param_groups[0]['lr']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.config.LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch, train_loss, val_loss, lr])
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}"):
            # 准备数据
            input_ids = batch["input_ids"].to(self.config.DEVICE)
            attention_mask = batch["attention_mask"].to(self.config.DEVICE)
            labels = batch["labels"].to(self.config.DEVICE)
            features = batch["features"].to(self.config.DEVICE)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask, labels, features)
            loss = outputs["loss"]
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.GRADIENT_CLIP
            )
            
            self.optimizer.step()
            
            total_loss += loss.item() * input_ids.size(0)
        
        # 计算平均训练损失
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.config.DEVICE)
                labels = batch["labels"].to(self.config.DEVICE)
                features = batch["features"].to(self.config.DEVICE)
                
                outputs = self.model(input_ids, attention_mask, labels, features)
                loss = outputs["loss"]
                
                total_loss += loss.item() * input_ids.size(0)
        
        # 计算平均验证损失
        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss
    
    def train(self):
        """完整训练流程"""
        print(f"开始训练，共 {self.config.EPOCHS} 个 epoch")
        
        for epoch in range(self.config.EPOCHS):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 打印指标
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 记录日志
            self._log_metrics(epoch+1, train_loss, val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"保存最佳模型至 {self.config.MODEL_SAVE_PATH}")
            
            # 调整学习率
            self.scheduler.step()
        
        print("训练完成!")
        return self.model
    