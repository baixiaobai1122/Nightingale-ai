# Medical Summarization Model Training Architecture

## 🎯 Training Objective

Train a medical summarization model using the medical_dialog dataset to generate:
1. **Clinician Summary**: Technical, structured summary for medical professionals
2. **Patient Summary**: Accessible, actionable summary for patients
3. **Provenance Tracking**: Maintain source attribution for all summary points

## 📊 Dataset Analysis: medical_dialog

### Dataset Structure
```
medical_dialog dataset from Hugging Face:
- **Size**: ~3.66M medical conversations
- **Languages**: English and Chinese versions available
- **Format**: JSON with dialogue structure
- **Fields**:
  - id: unique identifier
  - conversation: list of utterances
  - utterance: {speaker: "doctor/patient", text: "..."}
  - disease_tag: medical condition category
  - summary: existing summary (if available)
```

### Data Characteristics
- **Conversation Length**: Varies from 3-50+ turns
- **Medical Domains**: General practice, specialists, symptom queries
- **Speaker Roles**: Patient questions, doctor responses
- **Quality**: Varies, requires cleaning and validation

## 🏗️ Training Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Medical Dialog Training Pipeline              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Dataset   │───▶│    Data     │───▶│   Model     │         │
│  │ Processing  │    │ Pipeline    │    │  Training   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ PHI Redaction│    │ Augmentation│    │ Evaluation  │         │
│  │ & Validation │    │ & Synthesis │    │ & Metrics   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
ml_training/
├── data/
│   ├── raw/                     # Original medical_dialog data
│   ├── processed/               # Cleaned and preprocessed data
│   ├── synthetic/               # Synthetic training data
│   └── splits/                  # Train/val/test splits
├── models/
│   ├── base/                    # Pretrained models (T5, BART, etc.)
│   ├── checkpoints/             # Training checkpoints
│   └── final/                   # Final trained models
├── src/
│   ├── data_processing/         # Data preprocessing pipeline
│   ├── model_architecture/      # Model definitions
│   ├── training/                # Training loops and utilities
│   ├── evaluation/              # Evaluation metrics and tools
│   └── inference/               # Model inference code
├── configs/                     # Training configurations
├── scripts/                     # Utility scripts
├── notebooks/                   # Data analysis and experiments
└── outputs/                     # Training logs and results
```

## 🔄 Data Processing Pipeline

### Stage 1: Data Acquisition & Cleaning
```python
# data_processing/load_dataset.py
from datasets import load_dataset
import json
import pandas as pd

def load_medical_dialog():
    """Load and initial processing of medical_dialog dataset."""
    dataset = load_dataset("medical_dialog", "processed")

    # Extract conversations with quality filtering
    conversations = []
    for example in dataset['train']:
        if len(example['conversation']) >= 3:  # Minimum viable conversation
            conversations.append({
                'id': example['id'],
                'conversation': example['conversation'],
                'disease_tag': example.get('disease_tag', 'unknown'),
                'original_summary': example.get('summary', None)
            })

    return conversations
```

### Stage 2: PHI Detection & Redaction
```python
# data_processing/phi_processor.py
from backend.redact import redact_text, get_phi_analysis
import random

def process_conversation_phi(conversation):
    """Apply PHI redaction to training conversations."""
    processed_turns = []
    phi_metadata = []

    for turn in conversation['conversation']:
        # Redact PHI from each turn
        redacted_text, mapping = redact_text(turn['text'])
        phi_analysis = get_phi_analysis(turn['text'])

        processed_turns.append({
            'speaker': turn['speaker'],
            'text_original': turn['text'],          # Keep for training validation
            'text_redacted': redacted_text,         # Use for model training
            'phi_detected': bool(mapping),
            'phi_types': list(phi_analysis.keys())
        })

        phi_metadata.append({
            'turn_idx': len(processed_turns) - 1,
            'phi_count': len(mapping),
            'phi_types': list(phi_analysis.keys())
        })

    return processed_turns, phi_metadata
```

### Stage 3: Conversation Segmentation
```python
# data_processing/conversation_segmenter.py
def segment_conversation(conversation, max_context_length=512):
    """Split long conversations into trainable segments."""
    segments = []
    current_segment = []
    current_length = 0

    for i, turn in enumerate(conversation):
        turn_length = len(turn['text_redacted'].split())

        if current_length + turn_length > max_context_length and current_segment:
            # Save current segment and start new one
            segments.append({
                'turns': current_segment.copy(),
                'start_idx': current_segment[0]['turn_idx'],
                'end_idx': current_segment[-1]['turn_idx'],
                'total_turns': len(current_segment)
            })
            current_segment = [turn]
            current_length = turn_length
        else:
            current_segment.append(turn)
            current_length += turn_length

    # Add final segment
    if current_segment:
        segments.append({
            'turns': current_segment.copy(),
            'start_idx': current_segment[0]['turn_idx'],
            'end_idx': current_segment[-1]['turn_idx'],
            'total_turns': len(current_segment)
        })

    return segments
```

## 🤖 Model Architecture

### Dual-Output Transformer Architecture
```python
# model_architecture/dual_summarizer.py
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class DualMedicalSummarizer(nn.Module):
    """
    Transformer model that generates both clinician and patient summaries
    with provenance tracking.
    """

    def __init__(self, model_name="t5-base", max_length=512):
        super().__init__()

        # Base transformer model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Task-specific heads
        self.clinician_head = nn.Linear(self.model.config.d_model, self.model.config.vocab_size)
        self.patient_head = nn.Linear(self.model.config.d_model, self.model.config.vocab_size)

        # Provenance attention mechanism
        self.provenance_attention = nn.MultiheadAttention(
            embed_dim=self.model.config.d_model,
            num_heads=8
        )

        self.max_length = max_length

    def forward(self, input_ids, attention_mask, target_type="both"):
        """
        Args:
            input_ids: Tokenized conversation
            attention_mask: Attention mask
            target_type: "clinician", "patient", or "both"
        """
        # Get encoder outputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state

        outputs = {}

        if target_type in ["clinician", "both"]:
            # Generate clinician summary
            clinician_output = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )

            outputs['clinician'] = {
                'hidden_states': clinician_output.last_hidden_state,
                'logits': self.clinician_head(clinician_output.last_hidden_state)
            }

        if target_type in ["patient", "both"]:
            # Generate patient summary
            patient_output = self.model.decoder(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )

            outputs['patient'] = {
                'hidden_states': patient_output.last_hidden_state,
                'logits': self.patient_head(patient_output.last_hidden_state)
            }

        # Generate provenance attention scores
        if len(outputs) > 0:
            # Compute attention between summary and source segments
            source_segments = self._segment_source_attention(hidden_states, attention_mask)
            outputs['provenance'] = source_segments

        return outputs

    def _segment_source_attention(self, hidden_states, attention_mask):
        """Compute attention scores for provenance tracking."""
        # Segment the hidden states by turns/sentences
        # Return attention weights for each summary token to source segments
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Simple segmentation based on special tokens or sentence boundaries
        segment_attention = torch.zeros(batch_size, seq_len, seq_len)

        # This would be more sophisticated in practice
        return segment_attention
```

### Training Configuration
```python
# configs/training_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "t5-base"
    max_input_length: int = 512
    max_output_length: int = 256

    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Dual-task loss weights
    clinician_loss_weight: float = 0.6
    patient_loss_weight: float = 0.4
    provenance_loss_weight: float = 0.2

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Logging and checkpointing
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
```

## 🎓 Training Strategy

### Multi-Task Learning Approach
```python
# training/trainer.py
class MedicalSummarizerTrainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizers and schedulers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )

        # Loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.provenance_criterion = nn.MSELoss()

    def compute_loss(self, outputs, targets):
        """Compute multi-task loss."""
        total_loss = 0

        if 'clinician' in outputs and 'clinician' in targets:
            clinician_loss = self.criterion(
                outputs['clinician']['logits'].view(-1, outputs['clinician']['logits'].size(-1)),
                targets['clinician'].view(-1)
            )
            total_loss += self.config.clinician_loss_weight * clinician_loss

        if 'patient' in outputs and 'patient' in targets:
            patient_loss = self.criterion(
                outputs['patient']['logits'].view(-1, outputs['patient']['logits'].size(-1)),
                targets['patient'].view(-1)
            )
            total_loss += self.config.patient_loss_weight * patient_loss

        if 'provenance' in outputs and 'provenance' in targets:
            provenance_loss = self.provenance_criterion(
                outputs['provenance'],
                targets['provenance']
            )
            total_loss += self.config.provenance_loss_weight * provenance_loss

        return total_loss

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_type="both"
            )

            # Compute loss
            loss = self.compute_loss(outputs, batch['targets'])

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if batch_idx % self.config.log_every == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)
```

### Data Augmentation Strategies
```python
# data_processing/augmentation.py
class MedicalDialogAugmenter:
    """Augment medical dialog data for better generalization."""

    def __init__(self):
        self.symptom_synonyms = {
            "headache": ["head pain", "cephalgia", "head discomfort"],
            "fever": ["temperature", "pyrexia", "elevated temperature"],
            # ... more medical synonyms
        }

    def augment_conversation(self, conversation):
        """Apply various augmentation techniques."""
        augmented = []

        # 1. Synonym replacement
        augmented.append(self._replace_medical_synonyms(conversation))

        # 2. Turn reordering (where appropriate)
        augmented.append(self._reorder_turns(conversation))

        # 3. Paraphrasing
        augmented.append(self._paraphrase_turns(conversation))

        return augmented

    def _replace_medical_synonyms(self, conversation):
        """Replace medical terms with synonyms."""
        # Implementation for medical synonym replacement
        pass

    def _reorder_turns(self, conversation):
        """Reorder non-critical conversation turns."""
        # Implementation for turn reordering
        pass

    def _paraphrase_turns(self, conversation):
        """Generate paraphrased versions of turns."""
        # Implementation for paraphrasing
        pass
```

## 📊 Evaluation Framework

### Evaluation Metrics
```python
# evaluation/metrics.py
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np

class MedicalSummaryEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def evaluate_summaries(self, generated, reference, provenance_data=None):
        """Comprehensive evaluation of generated summaries."""
        metrics = {}

        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated)
        metrics.update({
            f'rouge_{k}': v.fmeasure for k, v in rouge_scores.items()
        })

        # BERTScore for semantic similarity
        P, R, F1 = score([generated], [reference], lang="en", verbose=False)
        metrics['bert_score_f1'] = F1.mean().item()

        # Medical-specific metrics
        metrics['medical_completeness'] = self._measure_medical_completeness(generated, reference)
        metrics['clinical_accuracy'] = self._measure_clinical_accuracy(generated)

        # Provenance accuracy
        if provenance_data:
            metrics['provenance_accuracy'] = self._measure_provenance_accuracy(
                generated, provenance_data
            )

        return metrics

    def _measure_medical_completeness(self, generated, reference):
        """Measure how well medical concepts are covered."""
        # Extract medical entities from both summaries
        # Compare coverage of medical concepts
        pass

    def _measure_clinical_accuracy(self, generated):
        """Assess clinical accuracy of generated summary."""
        # Use medical knowledge base to validate claims
        pass

    def _measure_provenance_accuracy(self, generated, provenance_data):
        """Measure accuracy of source attribution."""
        # Verify that claims in summary can be traced to source
        pass
```

### Human Evaluation Framework
```python
# evaluation/human_eval.py
class HumanEvaluationFramework:
    """Framework for human evaluation of medical summaries."""

    def __init__(self):
        self.evaluation_criteria = {
            'clinical_accuracy': {
                'scale': 1-5,
                'description': 'Medical accuracy and correctness'
            },
            'completeness': {
                'scale': 1-5,
                'description': 'Coverage of important medical information'
            },
            'readability': {
                'scale': 1-5,
                'description': 'Clarity and readability for target audience'
            },
            'provenance': {
                'scale': 1-5,
                'description': 'Accuracy of source attribution'
            }
        }

    def generate_evaluation_interface(self, summary_pairs):
        """Generate interface for human evaluators."""
        # Create evaluation forms for clinician and patient summaries
        pass

    def analyze_human_scores(self, evaluations):
        """Analyze inter-rater agreement and overall scores."""
        # Calculate Cohen's kappa, average scores, etc.
        pass
```

## 🚀 Training Pipeline Execution

### Training Script
```python
# scripts/train_model.py
#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig.from_file(args.config)

    # Setup data pipeline
    data_processor = MedicalDialogProcessor(args.data_path)
    train_loader, val_loader, test_loader = data_processor.get_dataloaders(config)

    # Initialize model
    model = DualMedicalSummarizer(config.model_name)

    # Setup trainer
    trainer = MedicalSummarizerTrainer(model, config, train_loader, val_loader)

    # Train model
    trainer.train()

    # Evaluate model
    evaluator = MedicalSummaryEvaluator()
    results = evaluator.evaluate(model, test_loader)

    print(f"Final Results: {results}")

if __name__ == "__main__":
    main()
```

## 📋 Implementation Timeline

### Phase 1: Data Pipeline (Week 1-2)
- [ ] Download and analyze medical_dialog dataset
- [ ] Implement PHI redaction for training data
- [ ] Create conversation segmentation pipeline
- [ ] Build train/val/test splits

### Phase 2: Model Architecture (Week 3-4)
- [ ] Implement dual summarization model
- [ ] Add provenance attention mechanism
- [ ] Create training configuration system
- [ ] Implement data augmentation

### Phase 3: Training Infrastructure (Week 5-6)
- [ ] Build training loop with multi-task loss
- [ ] Implement evaluation metrics
- [ ] Create checkpointing and logging
- [ ] Setup distributed training

### Phase 4: Evaluation & Optimization (Week 7-8)
- [ ] Train baseline models
- [ ] Implement human evaluation framework
- [ ] Optimize hyperparameters
- [ ] Generate final model for deployment

## 🎯 Success Metrics

### Technical Metrics
- **ROUGE-L > 0.45** for both summary types
- **BERTScore F1 > 0.85** for semantic similarity
- **Provenance Accuracy > 90%** for source attribution
- **PHI Leakage = 0%** in generated summaries

### Medical Quality Metrics
- **Clinical Accuracy > 4.0/5.0** (human evaluation)
- **Medical Completeness > 4.0/5.0** (human evaluation)
- **Readability Score** appropriate for target audience
- **Inter-rater Agreement κ > 0.7** for human evaluations

