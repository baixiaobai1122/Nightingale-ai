# 🏥 Medical Summarization Model Training

A comprehensive training pipeline for dual medical summarization using the UCSD medical_dialog dataset.

## 📋 Overview

This module trains AI models to generate two types of medical summaries from patient-doctor conversations:
- **Clinician Summary**: Technical, structured format for medical professionals
- **Patient Summary**: Accessible, friendly language for patients
- **Provenance Tracking**: All summary points include source anchors `[S#]` for traceability

## 🗂️ Project Structure

```
summary_training/
├── README.md                   # This file
├── train_medical_summarizer.py # Main training script
├── deploy_model.py            # Model deployment script
├── training_architecture.md   # Detailed architecture documentation
├── configs/
│   └── base_config.yaml       # Training configuration
├── src/
│   ├── data_processing/       # Data preprocessing pipeline
│   │   └── dataset_loader.py  # UCSD medical_dialog loader
│   ├── model_architecture/    # Model definitions
│   │   └── dual_summarizer.py # Dual-head T5 model
│   ├── training/              # Training framework
│   │   └── trainer.py         # Medical-specific trainer
│   └── inference/             # Model inference
│       └── model_inference.py # Production inference interface
├── data/                      # Data storage (created during training)
├── outputs/                   # Training outputs (created during training)
└── models/                    # Model checkpoints (created during training)
```

## 📊 Dataset Information

**Dataset**: [UCSD-DBMI/medical_dialog](https://huggingface.co/datasets/UCSD26/medical_dialog)
- **Size**: 260,000+ English medical conversations
- **Source**: healthcaremagic.com, icliniq.com
- **Coverage**: 96+ medical specialties
- **Format**: Doctor-patient dialogue with disease tags
- **Maintenance**: Actively maintained by UCSD

## 🚀 Quick Start

### Prerequisites

```bash
# Hardware Requirements
GPU: NVIDIA RTX 3080/4080 or Tesla V100 (≥12GB VRAM)
RAM: 32GB+
Storage: 100GB+ SSD

# Software Requirements
Python: 3.9+
CUDA: 11.8+
```

### Environment Setup

```bash
# 1. Create conda environment
conda create -n medical_summary python=3.9
conda activate medical_summary

# 2. Install core dependencies
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.33.0
pip install datasets==2.14.4
pip install accelerate==0.22.0
pip install wandb==0.15.8

# 3. Install evaluation dependencies
pip install rouge-score==0.1.2
pip install nltk==3.8.1
pip install evaluate==0.4.0

# 4. Install development tools
pip install jupyter notebook tensorboard
pip install pandas numpy matplotlib seaborn
```

### Data Pipeline Test

```bash
cd /home/chuc0007/mini

# Test dataset access
python -c "
from datasets import load_dataset
print('Testing dataset access...')
try:
    dataset = load_dataset('UCSD-DBMI/medical_dialog', 'processed', split='train[:100]')
    print(f'✅ Successfully loaded {len(dataset)} samples')
except Exception as e:
    print(f'❌ Error: {e}')
"

# Test data preprocessing pipeline
python -c "
import sys
sys.path.append('.')
from summary_training.src.data_processing.dataset_loader import MedicalDialogDataModule

config = {
    'cache_dir': './data/cache',
    'max_context_length': 512,
    'tokenizer_name': 't5-base'
}

print('=== Testing Data Pipeline ===')
data_module = MedicalDialogDataModule(config)
train_data, val_data, test_data = data_module.prepare_data()

print(f'Dataset sizes:')
print(f'  Train: {len(train_data)}')
print(f'  Val: {len(val_data)}')
print(f'  Test: {len(test_data)}')
print('✅ Data pipeline working correctly!')
"
```

## 🎯 Training Process

### Phase 1: Quick Test (10 minutes)

```bash
# Minimal training test to verify everything works
python summary_training/train_medical_summarizer.py \
    --model t5-small \
    --batch-size 2 \
    --epochs 1 \
    --max-source-length 256 \
    --max-target-length 128 \
    --output-dir ./outputs/test_run \
    --experiment-name "quick_test"

# Check outputs
ls -la ./outputs/test_run/
```

### Phase 2: Full Training (3-6 hours)

```bash
# Option 1: Using configuration file (recommended)
python summary_training/train_medical_summarizer.py \
    --config summary_training/configs/base_config.yaml

# Option 2: Command line parameters
python summary_training/train_medical_summarizer.py \
    --model t5-base \
    --epochs 15 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --use-wandb \
    --mixed-precision \
    --output-dir ./outputs/medical_summarizer_prod \
    --experiment-name "medical_summarizer_production"
```

### Phase 3: Training Monitoring

```bash
# Option 1: Weights & Biases (recommended)
wandb login  # Register at https://wandb.ai/
# Metrics automatically sync during training

# Option 2: TensorBoard
tensorboard --logdir ./outputs/medical_summarizer_v1/logs --port 6006

# Option 3: Log files
tail -f ./outputs/medical_summarizer_v1/training.log
```

### Phase 4: Resume Training (if interrupted)

```bash
python summary_training/train_medical_summarizer.py \
    --config summary_training/configs/base_config.yaml \
    --resume-from ./outputs/medical_summarizer_v1/checkpoints/checkpoint-5000
```

## 📈 Model Evaluation

### Performance Metrics

The training automatically evaluates:
- **ROUGE-1/2/L**: Summary quality scores
- **Medical Term Coverage**: Domain-specific vocabulary retention
- **Provenance Accuracy**: Source anchor correctness
- **Dual Summary Divergence**: Clinician vs patient style differences

### Manual Evaluation

```bash
# Test model inference after training
python -c "
import sys
sys.path.append('.')
from summary_training.src.inference.model_inference import load_inference_model

# Load trained model
model = load_inference_model('./outputs/medical_summarizer_v1/best_model')

# Test data
test_spans = [
    (1, 'Patient reports chest pain for 2 days'),
    (2, 'Pain worsens with activity'),
    (3, 'No previous cardiac history'),
    (4, 'Prescribed rest and follow-up')
]

# Generate summaries
clinician_summary, patient_summary = model.make_dual_summaries(test_spans)

print('=== AI-Generated Summaries ===')
print('📋 Clinician Summary:')
print(clinician_summary)
print('\\n👤 Patient Summary:')
print(patient_summary)
"
```

## 🚀 Model Deployment

### Deploy to Production Backend

```bash
# Deploy trained model to replace rule-based backend
python summary_training/deploy_model.py
```

This script will:
1. ✅ Verify trained model exists
2. 🔄 Backup current `backend/summarize.py`
3. 🤖 Replace with AI-powered implementation
4. 🧪 Test the new system
5. 📦 Enable hot-swapping between AI and rule-based systems

### Integration Architecture

```python
# Before deployment: Rule-based
def make_dual_summaries(spans):
    # Simple template-based generation
    return clinician_template, patient_template

# After deployment: AI-powered with fallback
def make_dual_summaries(spans):
    try:
        return ai_model.generate(spans)  # AI generation
    except:
        return template_fallback(spans)  # Rule-based fallback
```

### Verify Deployment

```bash
# Test the updated backend
cd /home/chuc0007/mini
python backend/tests/test_summary.py
python backend/tests/test_grounding.py
python backend/tests/test_latency.py

# Start backend service
python backend/app.py
```

## ⚙️ Configuration

### Training Configuration (`configs/base_config.yaml`)

```yaml
# Model Configuration
model_name: "t5-base"              # Base transformer model
max_source_length: 512             # Input conversation length
max_target_length: 256             # Output summary length

# Training Configuration
batch_size: 8                      # Training batch size
learning_rate: 5e-5                # Learning rate
num_epochs: 15                     # Training epochs
warmup_steps: 1000                 # Learning rate warmup

# Loss Weights
clinician_weight: 0.6              # Clinician summary loss weight
patient_weight: 0.4                # Patient summary loss weight

# Hardware
mixed_precision: true              # Use mixed precision training
device: "auto"                     # Auto-detect GPU/CPU
```

### Command Line Options

```bash
python train_medical_summarizer.py --help

# Key options:
--model          # Model name (t5-base, t5-large, bart-base)
--batch-size     # Training batch size
--epochs         # Number of training epochs
--learning-rate  # Learning rate
--use-wandb      # Enable Weights & Biases logging
--mixed-precision # Enable mixed precision training
--output-dir     # Output directory for checkpoints
--config         # YAML configuration file
```

## 🔍 Troubleshooting

### Common Issues

#### GPU Out of Memory
```bash
# Reduce batch size
--batch-size 4

# Use gradient accumulation
--gradient-accumulation-steps 2

# Use smaller model
--model t5-small
```

#### Dataset Download Issues
```bash
# Clear cache and retry
rm -rf ./data/cache
python summary_training/train_medical_summarizer.py --config configs/base_config.yaml
```

#### Training Interruption
```bash
# Resume from latest checkpoint
python summary_training/train_medical_summarizer.py \
    --config configs/base_config.yaml \
    --resume-from ./outputs/medical_summarizer_v1/checkpoints/checkpoint-latest
```

### Performance Optimization

#### For Limited GPU Memory:
- Use `t5-small` instead of `t5-base`
- Reduce `batch_size` to 4 or 2
- Enable `gradient_accumulation_steps: 4`

#### For Faster Training:
- Use multiple GPUs with `accelerate launch`
- Enable `mixed_precision: true`
- Increase `batch_size` if memory allows

## 📚 Additional Resources

- **Architecture Details**: [training_architecture.md](training_architecture.md)
- **Dataset Paper**: [MedDialog: Large-scale Medical Dialogue Datasets](https://aclanthology.org/2020.emnlp-main.743/)
- **Hugging Face Dataset**: [UCSD26/medical_dialog](https://huggingface.co/datasets/UCSD26/medical_dialog)
- **Transformer Models**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 🤝 Contributing

This training pipeline integrates with the larger Nightingale AI medical assistant system. Key integration points:

- **PHI Redaction**: Uses `backend.redact` for HIPAA-compliant data processing
- **Backend Integration**: Seamlessly replaces `backend.summarize.make_dual_summaries()`
- **Frontend Compatible**: Maintains existing API contracts
- **Security**: Inherits all security and audit features

## 📄 License

This project is part of the Nightingale AI medical assistant system. Please refer to the main project license for usage terms.

---

**🎯 Ready to train your medical summarization model? Start with the Quick Test phase above!**