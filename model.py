# 模型架构模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

class LoRALayer(nn.Module):
    """低秩适配层，用于参数高效微调"""
    def __init__(self, in_features, out_features, rank=32, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.W_a = nn.Linear(in_features, rank, bias=False)
        self.W_b = nn.Linear(rank, out_features, bias=False)
        
        # 初始化权重
        nn.init.normal_(self.W_a.weight, std=0.02)
        nn.init.zeros_(self.W_b.weight)
        
    def forward(self, x):
        x = self.W_a(x)
        x = self.W_b(x)
        return x * self.scaling


class PIIExpert(nn.Module):
    """PII语义专家：处理PII衍生密码"""
    def __init__(self, base_model, hidden_dim=1024, rank=32):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # 添加BiLSTM层处理结构化PII
        self.bilstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # 贝叶斯先验层
        self.bayesian_prior = nn.Linear(hidden_dim, self.config.vocab_size)
        
        # LoRA适配层
        self.lora = LoRALayer(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            rank=rank
        )
        
    def forward(self, input_ids, attention_mask, features=None):
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]  # 最后一层的隐藏状态
        
        # 应用LoRA
        lora_output = self.lora(hidden_states)
        hidden_states = hidden_states + lora_output
        
        # BiLSTM处理
        lstm_out, _ = self.bilstm(hidden_states)
        
        # 贝叶斯先验层
        logits = self.bayesian_prior(lstm_out)
        
        return logits


class HighEntropyExpert(nn.Module):
    """高熵专家：生成高熵随机密码"""
    def __init__(self, base_model, rank=32):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # LoRA适配层
        self.lora = LoRALayer(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            rank=rank
        )
        
        # 变分dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 输出层
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        
    def forward(self, input_ids, attention_mask, features=None):
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        
        # 应用LoRA
        lora_output = self.lora(hidden_states)
        hidden_states = hidden_states + lora_output
        
        # 应用变分dropout增加随机性
        hidden_states = self.dropout(hidden_states)
        
        # 输出层
        logits = self.output_layer(hidden_states)
        
        # 如果提供了特征，应用熵自适应调整
        if features is not None:
            entropy = features[:, 2].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            # 基于熵调整logits，增加高熵区域的随机性
            scale = 1.0 + 0.5 * entropy  # 熵越高，缩放越大
            logits = logits * scale
        
        return logits


class LeetSpeakExpert(nn.Module):
    """词形转换专家：处理Leetspeak等转换密码"""
    def __init__(self, base_model, leet_dictionary, rank=32):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.leet_dictionary = leet_dictionary
        self.tokenizer = None  # 稍后设置
        
        # LoRA适配层
        self.lora = LoRALayer(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            rank=rank
        )
        
        # 输出层
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        
    def forward(self, input_ids, attention_mask, features=None):
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        
        # 应用LoRA
        lora_output = self.lora(hidden_states)
        hidden_states = hidden_states + lora_output
        
        # 输出层
        logits = self.output_layer(hidden_states)
        
        return logits
    
    def leet_similarity_loss(self, generated_ids, target_ids):
        """计算与Leetspeak的相似度损失"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set for LeetSpeakExpert")
            
        batch_size, seq_len = generated_ids.shape
        loss = 0.0
        
        for i in range(batch_size):
            generated = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            target = self.tokenizer.decode(target_ids[i], skip_special_tokens=True)
            
            # 计算Levenshtein距离
            lev_dist = self._levenshtein_distance(generated, target)
            loss += lev_dist / max(len(generated), len(target), 1)
            
        return loss / batch_size
    
    def _levenshtein_distance(self, s1, s2):
        """计算两个字符串之间的Levenshtein距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]


class GatingNetwork(nn.Module):
    """语义感知门控网络，用于动态选择专家"""
    def __init__(self, input_dim=3, hidden_dim=64, num_experts=3):
        super().__init__()
        # 混合CNN-GRU架构
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, features):
        # 特征形状: [batch_size, input_dim]
        # 调整形状适应CNN: [batch_size, input_dim, 1]
        x = features.unsqueeze(2)
        
        # CNN处理
        x = self.cnn(x)  # [batch_size, hidden_dim, 1]
        x = x.squeeze(2)  # [batch_size, hidden_dim]
        
        # GRU处理
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x, _ = self.gru(x)
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # 输出专家权重
        logits = self.fc(x)
        weights = self.softmax(logits)
        
        return weights


class PassMoEP(nn.Module):
    """完整的PassMoE-P模型"""
    def __init__(self, base_model_name, leet_dictionary, config):
        super().__init__()
        # 加载基础模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 配置参数
        self.config = config
        
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 创建三个专家
        self.experts = nn.ModuleList([
            PIIExpert(self.base_model, hidden_dim=config.HIDDEN_DIM, rank=config.LORA_RANK),
            HighEntropyExpert(self.base_model, rank=config.LORA_RANK),
            LeetSpeakExpert(self.base_model, leet_dictionary, rank=config.LORA_RANK)
        ])
        
        # 设置LeetSpeakExpert的tokenizer
        self.experts[2].tokenizer = self.tokenizer
        
        # 创建门控网络
        self.gating_network = GatingNetwork()
        
        # Leetspeak字典
        self.leet_dictionary = leet_dictionary
        
        # 梯度隔离掩码
        self.gradient_mask = None
        
    def set_gradient_mask(self, mask):
        """设置梯度隔离掩码，控制哪些专家的梯度会被更新"""
        self.gradient_mask = mask
        
    def forward(self, input_ids, attention_mask, labels, features):
        # 获取专家权重 [batch_size, 3]
        expert_weights = self.gating_network(features)
        
        # 选择Top-2专家
        top2_weights, top2_indices = torch.topk(expert_weights, k=2, dim=1)
        
        # 归一化权重
        top2_weights = F.softmax(top2_weights, dim=1)
        
        # 计算每个专家的输出
        expert_logits = []
        for i, expert in enumerate(self.experts):
            logits = expert(input_ids, attention_mask, features)
            expert_logits.append(logits)
        
        # 计算加权输出
        batch_size = input_ids.size(0)
        final_logits = torch.zeros_like(expert_logits[0])
        
        for i in range(batch_size):
            # 获取当前样本的Top-2专家索引和权重
            idx1, idx2 = top2_indices[i]
            w1, w2 = top2_weights[i]
            
            # 加权组合
            final_logits[i] = w1 * expert_logits[idx1][i] + w2 * expert_logits[idx2][i]
        
        # 计算交叉熵损失
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        ce_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 为词形转换专家添加Levenshtein损失
        leet_loss = 0.0
        
        # 获取生成的token（取logits最大的）
        generated_ids = torch.argmax(final_logits, dim=-1)
        
        # 只为激活了词形转换专家的样本计算额外损失
        leet_expert_idx = 2
        leet_mask = (top2_indices == leet_expert_idx).any(dim=1)
        
        if leet_mask.any() and self.config.LEET_LOSS_LAMBDA > 0:
            leet_samples = generated_ids[leet_mask]
            leet_targets = labels[leet_mask]
            leet_loss = self.experts[leet_expert_idx].leet_similarity_loss(leet_samples, leet_targets)
            leet_loss = self.config.LEET_LOSS_LAMBDA * leet_loss
        
        # 总损失
        total_loss = ce_loss + leet_loss
        
        # 梯度隔离：只更新被选中专家的梯度
        if self.training and self.gradient_mask is not None:
            for i, expert in enumerate(self.experts):
                for param in expert.parameters():
                    if param.grad is not None and not self.gradient_mask[i]:
                        param.grad = None
        
        return {
            "loss": total_loss,
            "logits": final_logits,
            "expert_weights": expert_weights
        }
    
    def generate_passwords(self, prefix="", num_passwords=10):
        """生成密码的函数"""
        self.eval()
        
        # 编码前缀
        input_ids = self.tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.base_model.device)
        
        # 提取前缀特征
        features = self._extract_features(prefix)
        features = torch.tensor([features], dtype=torch.float32).to(self.base_model.device)
        
        generated_passwords = []
        queue = [(input_ids, 1.0)]  # (当前序列, 累积概率)
        
        with torch.no_grad():
            while queue and len(generated_passwords) < num_passwords:
                current_ids, current_prob = queue.pop(0)
                
                # 如果达到最大长度，跳过
                if current_ids.size(1) >= self.config.MAX_LENGTH:
                    continue
                
                # 获取当前特征
                current_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
                current_features = self._extract_features(current_text)
                current_features = torch.tensor([current_features], dtype=torch.float32).to(self.base_model.device)
                
                # 获取专家权重
                expert_weights = self.gating_network(current_features)
                top2_weights, top2_indices = torch.topk(expert_weights, k=2, dim=1)
                top2_weights = F.softmax(top2_weights, dim=1)
                
                # 获取专家输出
                logits = []
                for idx in top2_indices[0]:
                    expert_logits = self.experts[idx](current_ids, torch.ones_like(current_ids), current_features)
                    logits.append(expert_logits)
                
                # 加权组合logits
                combined_logits = top2_weights[0, 0] * logits[0] + top2_weights[0, 1] * logits[1]
                
                # 最后一个token的logits
                next_token_logits = combined_logits[:, -1, :] / self.config.TEMPERATURE
                
                # 应用softmax
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # 采样多个候选token
                top_probs, top_indices = torch.topk(next_token_probs, self.config.NUM_CANDIDATES)
                
                for i in range(self.config.NUM_CANDIDATES):
                    token_id = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    token_prob = top_probs[0, i].item()
                    new_prob = current_prob * token_prob
                    
                    # 拼接新token
                    new_ids = torch.cat([current_ids, token_id], dim=1)
                    
                    # 检查是否是结束符
                    if token_id == self.tokenizer.eos_token_id:
                        password = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
                        if new_prob > self.config.TAU:  # 概率阈值筛选
                            generated_passwords.append((password, new_prob))
                    else:
                        if new_prob > self.config.TAU / 10:  # 放宽前缀的概率阈值
                            queue.append((new_ids, new_prob))
                
                # 对队列按概率排序，优先处理高概率序列
                queue.sort(key=lambda x: x[1], reverse=True)
        
        # 去重并按概率排序
        unique_passwords = {}
        for pwd, prob in generated_passwords:
            if pwd not in unique_passwords or prob > unique_passwords[pwd]:
                unique_passwords[pwd] = prob
        
        # 转换为列表并排序
        result = sorted(unique_passwords.items(), key=lambda x: x[1], reverse=True)
        return [pwd for pwd, _ in result[:num_passwords]]
    
    def _extract_features(self, text):
        """提取文本特征，与数据集的方法一致"""
        pii_score = self._detect_pii(text)
        leet_score = self._detect_leetspeak(text)
        entropy = self._calculate_entropy(text)
        return [pii_score, leet_score, entropy]
    
    def _detect_pii(self, text):
        year_patterns = [r'\b19\d{2}\b', r'\b20\d{2}\b']
        name_fragments = ['john', 'mary', 'zhang', 'li', 'wang']
        
        score = 0.0
        for pattern in year_patterns:
            if re.search(pattern, text):
                score += 0.3
                
        for fragment in name_fragments:
            if fragment in text.lower():
                score += 0.3
                
        return min(score, 1.0)
    
    def _detect_leetspeak(self, text):
        leet_chars = set()
        for mappings in self.leet_dictionary.values():
            leet_chars.update(mappings)
            
        count = 0
        for char in text:
            if char in leet_chars:
                count += 1
                
        return min(count / max(len(text), 1), 1.0)
    
    def _calculate_entropy(self, text):
        if not text:
            return 0.0
        
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        entropy = -sum(p * math.log2(p) for p in prob) if prob else 0.0
        return min(entropy / 8.0, 1.0)
    