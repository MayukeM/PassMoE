# PassMoE-P: A Pattern-Specialized Mixture-of-Experts Framework for Password Generation

PassMoE-P (Password Generation with Pattern-specialized Mixture-of-Experts) is the first pattern-specialized mixture-of-experts framework designed for password guessing tasks, achieving a Pareto-optimal balance between efficiency and effectiveness in password generation through semantic-level task decomposition.

## Project Overview

PassMoE-P addresses the scalability bottleneck of monolithic Large Language Models (LLMs) in password guessing caused by heterogeneous password patterns. Its core innovation integrates behavioral semantic domain knowledge into three parameter-efficient LoRA experts, with dynamic routing via a hybrid CNN-GRU gating network based on quantized semantic features.

Evaluated on a heterogeneous dataset covering 70.5 million real-world passwords, the framework not only significantly outperforms existing state-of-the-art baselines like PassLLM and PassGPT in attack success rate but also reduces VRAM consumption by 50% and improves throughput by 4.8x.

## Architecture Features


1. **Shared LLM Backbone**: Built on pre-trained language models (Qwen2.5/Llama3/Mistral) providing fundamental language understanding capabilities.
   
2. **Three Pattern-Specialized LoRA Experts**:
   - PII Semantic Expert: Handles passwords derived from personal information (names, birthdays, etc.)
   - High-Entropy Expert: Generates complex passwords with high randomness
   - Lexical Transformation Expert: Processes character transformation patterns like Leetspeak

3. **Semantic-Aware Gating Network**:
   - Hybrid CNN-GRU architecture extracting password features (PII score, Leetspeak score, structural entropy)
   - Top-2 routing strategy dynamically selecting optimal expert combinations

4. **Parameter-Efficient Design**: Utilizes LoRA technology to reduce trainable parameters by 98.6%, enabling efficient fine-tuning

## Quick Start

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (recommended for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/MayukeM/PassMoE.git
cd PassMoE

# Install dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies
pip install -r requirements-dev.txt
```

### Basic Usage

#### 1. Training the Model

```bash
# Train with default configuration
python main.py

# Train with custom parameters
python main.py --epochs 15 --batch_size 32 --base_model "meta-llama/Llama-3-8B"
```

#### 2. Generating Passwords

```python
from model import PassMoEP
from config import Config, LEET_DICTIONARY
import torch

# Load configuration and model
config = Config()
model = PassMoEP(
    base_model_name=config.BASE_MODEL_NAME,
    leet_dictionary=LEET_DICTIONARY,
    config=config
)
model.load_state_dict(torch.load("models/best_passmoe_p.pt"))

# Generate passwords
generated = model.generate_passwords(prefix="", num_passwords=10)
print("Generated passwords:")
for i, pwd in enumerate(generated, 1):
    print(f"{i}. {pwd}")

# Generate passwords with prefix
generated = model.generate_passwords(prefix="Zhang", num_passwords=5)
```

#### 3. Evaluating the Model

```bash
# Evaluate model performance
python evaluate.py --model_path "models/best_passmoe_p.pt" --test_data "data/test_passwords.csv"
```

## Configuration

All configurable parameters are centralized in `config.py`, including:

- **Model Parameters**: Base model selection, LoRA rank, hidden dimension, etc.
- **Training Parameters**: Learning rate, batch size, number of epochs, gradient clipping threshold, etc.
- **Data Parameters**: Maximum password length, dataset paths, validation split ratio, etc.
- **Generation Parameters**: Temperature coefficient, beam search width, probability threshold, etc.

## Project Structure

```
PassMoE/
├── config.py           # Configuration parameters
├── data.py             # Data loading and preprocessing
├── model.py            # Model architecture implementation
├── trainer.py          # Training logic
├── main.py             # Main program entry
├── evaluate.py         # Model evaluation tools
├── requirements.txt    # Dependencies
├── requirements-dev.txt # Development dependencies
├── data/               # Data directory
├── models/             # Model saving directory
└── results/            # Evaluation results directory
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please contact: 1374079897@qq.com
