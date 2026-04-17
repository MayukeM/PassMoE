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

## Ethical Considerations
PassMoE is a project focused on password security optimization based on Mixture-of-Experts (MoE) models. Given its relevance to user privacy, data security, and potential deployment in sensitive scenarios (e.g., personal identity verification, enterprise data protection), ethical considerations are integral to the project’s development and deployment. This section outlines the core ethical principles, key risks, mitigation strategies, and compliance commitments of PassMoE.

### 1. Core Ethical Principles
The development and maintenance of PassMoE adhere to the following non-negotiable ethical principles:
- **Privacy First**: User passwords, credentials, or sensitive data processed by PassMoE are never collected, stored, or shared without explicit, informed consent from the user.
- **Non-Maleficence**: The project is designed to enhance security and usability; all features are built to prevent harm (e.g., unauthorized access, data breaches, or misuse for malicious purposes).
- **Transparency**: Technical limitations, potential risks, and intended use cases of PassMoE are clearly documented to avoid user misunderstanding or misapplication.
- **Fairness**: The tool/algorithm is accessible to all legitimate users, with no discrimination based on region, gender, socioeconomic status, or linguistic background.
- **Accountability**: Project maintainers take responsibility for addressing ethical breaches (e.g., vulnerability exploitation) and provide timely updates to mitigate risks.

### 2. Key Ethical Risks & Mitigation Strategies
| Risk Category               | Risk Description                                                                 | Mitigation Strategies                                                                 |
|------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Data Privacy & Security      | Unauthorized access to PassMoE-processed data may lead to password leaks, identity theft, or financial losses for users. | - Implement end-to-end encryption for all user data processed by the model;<br>- Prohibit storage of plaintext passwords (only store salted/hased values);<br>- Conduct quarterly security audits to identify and fix vulnerabilities;<br>- Add role-based access control (RBAC) for admin/developer access to project infrastructure. |
| Technical Misuse             | Malicious actors may exploit PassMoE’s algorithms to crack weak passwords or bypass legitimate security systems. | - Explicitly prohibit illegal use in project documentation and license agreements;<br>- Integrate anomaly detection for suspicious usage (e.g., frequent brute-force attempts);<br>- Avoid open-sourcing weaponizable core modules (or restrict access to sensitive code via approval workflows);<br>- Collaborate with cybersecurity communities to report and address misuse cases. |
| Algorithmic Bias/Fairness    | The password strength evaluation algorithm may be biased (e.g., penalizing non-English character sets), disproportionately affecting non-Western users. | - Test the algorithm with multilingual password datasets (Latin, Cyrillic, Chinese, Arabic, etc.);<br>- Adjust scoring logic to avoid cultural/linguistic bias;<br>- Document known limitations for non-English use cases and commit to iterative optimization. |
| Liability & Transparency     | Users may misattribute security failures (e.g., password leaks) to PassMoE, or lack clarity on responsibility for technical issues. | - Publish a clear liability disclaimer (distinguish PassMoE’s responsibility from user misconfiguration);<br>- Maintain a public GitHub Issues tracker to address ethical/security concerns transparently;<br>- Disclose technical limitations (e.g., "PassMoE does not guarantee 100% protection against advanced brute-force attacks"). |

### 3. Compliance with Laws & Regulations
PassMoE strictly complies with global and regional data protection and cybersecurity regulations, including:
- **GDPR (EU)**: Uphold user rights to data access, deletion, and consent withdrawal for all EU-based users.
- **CCPA/CPRA (California, US)**: Ensure transparency in data processing and opt-out options for California consumers.
- **Cybersecurity Laws (China)**: Adhere to data localization and security assessment requirements for Chinese users.
- **Industry Standards**: Align with NIST Digital Identity Guidelines (SP 800-63B) for password management and ISO/IEC 27001 for information security.
All code and documentation comply with open-source license terms (MIT) and intellectual property laws (no use of proprietary algorithms without authorization).

### 4. Ongoing Ethical Review & Improvement
Ethical considerations are not one-time checks but an ongoing process for PassMoE:
- Conduct bi-annual ethical reviews involving external cybersecurity and ethics experts to identify emerging risks (e.g., new attack vectors targeting MoE-based password systems).
- Collect and respond to user feedback on ethical concerns via GitHub Discussions or dedicated email channels.
- Update this ethical considerations section alongside major version releases (e.g., v2.0) to reflect new features or risk scenarios.
- Provide contributor guidelines that enforce ethical coding practices (e.g., avoiding bias in algorithm design, rejecting malicious pull requests).

### 5. Conclusion
Ethics is embedded in every stage of PassMoE’s lifecycle—from code design to documentation and deployment. The project prioritizes user privacy, security, and fairness, and is committed to mitigating potential harms while maximizing societal benefits. We welcome feedback from the community to continuously improve the ethical alignment of PassMoE and ensure it serves as a responsible tool for password security enhancement.

## Contact

For questions or suggestions, please contact: 1374079897@qq.com
