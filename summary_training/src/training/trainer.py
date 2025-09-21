"""
Training Framework for Dual Medical Summarization Model
Includes multi-task learning, evaluation, and medical-specific metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

import numpy as np
import wandb
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import time
from tqdm import tqdm

# Evaluation metrics
from rouge_score import rouge_scorer
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("BERTScore not available. Install with: pip install bert-score")

from ..model_architecture.dual_summarizer import DualMedicalSummarizer
from ..data_processing.dataset_loader import MedicalDialogDataModule

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration for dual medical summarizer."""

    # Model settings
    model_name: str = "t5-base"
    max_source_length: int = 512
    max_target_length: int = 256
    freeze_encoder: bool = False

    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Multi-task loss weights
    clinician_weight: float = 0.6
    patient_weight: float = 0.4
    provenance_weight: float = 0.2

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Data settings
    data_cache_dir: str = "./data/cache"
    max_context_length: int = 512

    # Logging and checkpointing
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500
    output_dir: str = "./outputs"
    experiment_name: str = "dual_medical_summarizer"

    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 4

    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "medical-summarization"

class MedicalSummaryEvaluator:
    """Comprehensive evaluation for medical summaries."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        # Medical term lists for evaluation
        self.medical_terms = self._load_medical_terms()

    def _load_medical_terms(self) -> Dict[str, List[str]]:
        """Load medical terminology for evaluation."""
        # This would load from a medical dictionary in practice
        return {
            "symptoms": ["pain", "fever", "nausea", "headache", "fatigue", "cough"],
            "body_parts": ["head", "chest", "abdomen", "back", "throat", "stomach"],
            "medications": ["ibuprofen", "acetaminophen", "antibiotic", "prescription"],
            "procedures": ["examination", "test", "scan", "blood work", "x-ray"]
        }

    def evaluate_summary_pair(
        self,
        clinician_generated: str,
        patient_generated: str,
        clinician_reference: Optional[str] = None,
        patient_reference: Optional[str] = None,
        source_text: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate both clinician and patient summaries."""

        metrics = {}

        # Evaluate clinician summary
        if clinician_reference:
            clin_metrics = self._evaluate_single_summary(
                clinician_generated, clinician_reference, "clinician"
            )
            metrics.update({f"clinician_{k}": v for k, v in clin_metrics.items()})

        # Evaluate patient summary
        if patient_reference:
            pat_metrics = self._evaluate_single_summary(
                patient_generated, patient_reference, "patient"
            )
            metrics.update({f"patient_{k}": v for k, v in pat_metrics.items()})

        # Cross-summary consistency
        metrics["summary_consistency"] = self._measure_consistency(
            clinician_generated, patient_generated
        )

        # Medical content evaluation
        if source_text:
            metrics.update(self._evaluate_medical_content(
                clinician_generated, patient_generated, source_text
            ))

        return metrics

    def _evaluate_single_summary(
        self, generated: str, reference: str, summary_type: str
    ) -> Dict[str, float]:
        """Evaluate a single summary against reference."""

        metrics = {}

        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated)
        for rouge_type, scores in rouge_scores.items():
            metrics[f"{rouge_type}_f1"] = scores.fmeasure
            metrics[f"{rouge_type}_precision"] = scores.precision
            metrics[f"{rouge_type}_recall"] = scores.recall

        # BERTScore (if available)
        if BERTSCORE_AVAILABLE:
            try:
                _, _, f1 = bert_score([generated], [reference], lang="en", verbose=False)
                metrics["bert_score_f1"] = f1.mean().item()
            except Exception as e:
                logger.warning(f"BERTScore computation failed: {e}")
                metrics["bert_score_f1"] = 0.0

        # Length-based metrics
        metrics["length_ratio"] = len(generated.split()) / max(len(reference.split()), 1)

        # Medical terminology coverage
        metrics["medical_term_coverage"] = self._measure_medical_coverage(generated)

        # Summary type appropriateness
        metrics["type_appropriateness"] = self._measure_type_appropriateness(
            generated, summary_type
        )

        return metrics

    def _measure_consistency(self, clinician_summary: str, patient_summary: str) -> float:
        """Measure consistency between clinician and patient summaries."""

        # Extract key medical facts from both summaries
        clin_facts = self._extract_medical_facts(clinician_summary)
        pat_facts = self._extract_medical_facts(patient_summary)

        if not clin_facts and not pat_facts:
            return 1.0

        # Calculate overlap
        common_facts = set(clin_facts) & set(pat_facts)
        all_facts = set(clin_facts) | set(pat_facts)

        return len(common_facts) / max(len(all_facts), 1)

    def _extract_medical_facts(self, text: str) -> List[str]:
        """Extract medical facts from summary text."""
        facts = []
        words = text.lower().split()

        # Simple extraction based on medical terms
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in words:
                    facts.append(f"{category}:{term}")

        return facts

    def _measure_medical_coverage(self, summary: str) -> float:
        """Measure coverage of medical terminology."""
        words = summary.lower().split()

        total_medical_terms = sum(len(terms) for terms in self.medical_terms.values())
        found_terms = 0

        for terms in self.medical_terms.values():
            for term in terms:
                if term in words:
                    found_terms += 1

        return found_terms / max(total_medical_terms, 1)

    def _measure_type_appropriateness(self, summary: str, summary_type: str) -> float:
        """Measure if summary matches expected style for target audience."""

        text_lower = summary.lower()

        if summary_type == "clinician":
            # Clinical summaries should have medical terminology
            clinical_indicators = [
                "patient presents", "diagnosis", "treatment", "clinical",
                "assessment", "examination", "symptoms", "condition"
            ]
            score = sum(1 for indicator in clinical_indicators if indicator in text_lower)
            return min(score / 3, 1.0)  # Normalize to [0, 1]

        elif summary_type == "patient":
            # Patient summaries should be more accessible
            patient_indicators = [
                "you", "your", "we discussed", "we talked about",
                "next steps", "follow up", "what this means"
            ]
            score = sum(1 for indicator in patient_indicators if indicator in text_lower)
            return min(score / 2, 1.0)  # Normalize to [0, 1]

        return 0.5  # Neutral score if type unknown

    def _evaluate_medical_content(
        self, clinician_summary: str, patient_summary: str, source_text: str
    ) -> Dict[str, float]:
        """Evaluate medical content accuracy and completeness."""

        metrics = {}

        # Extract medical content from source
        source_facts = self._extract_medical_facts(source_text)
        clin_facts = self._extract_medical_facts(clinician_summary)
        pat_facts = self._extract_medical_facts(patient_summary)

        # Content coverage
        if source_facts:
            clin_coverage = len(set(clin_facts) & set(source_facts)) / len(source_facts)
            pat_coverage = len(set(pat_facts) & set(source_facts)) / len(source_facts)

            metrics["clinician_content_coverage"] = clin_coverage
            metrics["patient_content_coverage"] = pat_coverage

        # Content preservation (no hallucination)
        all_source_facts = set(source_facts)
        clin_hallucination = len(set(clin_facts) - all_source_facts) / max(len(clin_facts), 1)
        pat_hallucination = len(set(pat_facts) - all_source_facts) / max(len(pat_facts), 1)

        metrics["clinician_hallucination_rate"] = clin_hallucination
        metrics["patient_hallucination_rate"] = pat_hallucination

        return metrics

class DualMedicalTrainer:
    """Trainer for dual medical summarization model."""

    def __init__(
        self,
        model: DualMedicalSummarizer,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup scheduler
        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        # Setup evaluator
        self.evaluator = MedicalSummaryEvaluator()

        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Setup logging
        self.setup_logging()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_score = -float('inf')

    def setup_logging(self):
        """Setup logging and experiment tracking."""

        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Setup Wandb
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )

    def train(self):
        """Main training loop."""

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total training steps: {len(self.train_loader) * self.config.num_epochs}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Training
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} training metrics: {train_metrics}")

            # Validation
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch + 1} validation metrics: {val_metrics}")

            # Save checkpoint
            if val_metrics['rouge_l_f1'] > self.best_val_score:
                self.best_val_score = val_metrics['rouge_l_f1']
                self.save_checkpoint(is_best=True)
                logger.info(f"New best model saved with ROUGE-L F1: {self.best_val_score:.4f}")

            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(is_best=False)

            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()}
                })

        logger.info("Training completed")

        # Final evaluation on test set
        if self.test_loader:
            test_metrics = self.evaluate_test_set()
            logger.info(f"Final test metrics: {test_metrics}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""

        self.model.train()
        total_loss = 0
        total_clinician_loss = 0
        total_patient_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        clinician_target_ids=batch['clinician_target_ids'],
                        patient_target_ids=batch['patient_target_ids'],
                        target_type="both"
                    )

                    # Compute losses
                    clinician_loss = outputs.get('clinician_loss', torch.tensor(0.0))
                    patient_loss = outputs.get('patient_loss', torch.tensor(0.0))

                    total_loss_batch = (
                        self.config.clinician_weight * clinician_loss +
                        self.config.patient_weight * patient_loss
                    )

                # Backward pass with mixed precision
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    clinician_target_ids=batch['clinician_target_ids'],
                    patient_target_ids=batch['patient_target_ids'],
                    target_type="both"
                )

                # Compute losses
                clinician_loss = outputs.get('clinician_loss', torch.tensor(0.0))
                patient_loss = outputs.get('patient_loss', torch.tensor(0.0))

                total_loss_batch = (
                    self.config.clinician_weight * clinician_loss +
                    self.config.patient_weight * patient_loss
                )

                # Backward pass
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_clinician_loss += clinician_loss.item()
            total_patient_loss += patient_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss_batch.item(),
                'clin_loss': clinician_loss.item(),
                'pat_loss': patient_loss.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })

            # Logging
            if self.global_step % self.config.log_every == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"loss={total_loss_batch.item():.4f}, "
                    f"clin_loss={clinician_loss.item():.4f}, "
                    f"pat_loss={patient_loss.item():.4f}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

        # Return average losses
        return {
            'total_loss': total_loss / num_batches,
            'clinician_loss': total_clinician_loss / num_batches,
            'patient_loss': total_patient_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""

        self.model.eval()
        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Generate summaries
                generated = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target_type="both"
                )

                # Decode target summaries for comparison
                clinician_targets = self.model.tokenizer.batch_decode(
                    batch['clinician_target_ids'], skip_special_tokens=True
                )
                patient_targets = self.model.tokenizer.batch_decode(
                    batch['patient_target_ids'], skip_special_tokens=True
                )

                # Evaluate each example in batch
                for i in range(len(generated['clinician_summaries'])):
                    metrics = self.evaluator.evaluate_summary_pair(
                        clinician_generated=generated['clinician_summaries'][i],
                        patient_generated=generated['patient_summaries'][i],
                        clinician_reference=clinician_targets[i],
                        patient_reference=patient_targets[i]
                    )
                    all_metrics.append(metrics)

        # Aggregate metrics
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated_metrics[key] = np.mean(values)

        return aggregated_metrics

    def evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate on test set."""
        if not self.test_loader:
            return {}

        # Load best model
        best_model_path = self.output_dir / "best_model.pt"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
            logger.info("Loaded best model for test evaluation")

        self.model.eval()
        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test Evaluation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Generate summaries
                generated = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target_type="both"
                )

                # Decode targets
                clinician_targets = self.model.tokenizer.batch_decode(
                    batch['clinician_target_ids'], skip_special_tokens=True
                )
                patient_targets = self.model.tokenizer.batch_decode(
                    batch['patient_target_ids'], skip_special_tokens=True
                )

                # Evaluate
                for i in range(len(generated['clinician_summaries'])):
                    metrics = self.evaluator.evaluate_summary_pair(
                        clinician_generated=generated['clinician_summaries'][i],
                        patient_generated=generated['patient_summaries'][i],
                        clinician_reference=clinician_targets[i],
                        patient_reference=patient_targets[i]
                    )
                    all_metrics.append(metrics)

        # Aggregate test metrics
        test_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            test_metrics[f"test_{key}"] = np.mean(values)

        # Save test results
        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump(test_metrics, f, indent=2)

        return test_metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config.__dict__
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_model_path = self.output_dir / "best_model.pt"
            torch.save(self.model.state_dict(), best_model_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

# Main training script
def main():
    """Main training function."""

    # Configuration
    config = TrainingConfig(
        model_name="t5-base",
        batch_size=8,
        learning_rate=5e-5,
        num_epochs=10,
        use_wandb=True,
        experiment_name="dual_medical_summarizer_v1"
    )

    # Setup data
    data_config = {
        'cache_dir': config.data_cache_dir,
        'max_context_length': config.max_context_length,
        'tokenizer_name': config.model_name
    }

    data_module = MedicalDialogDataModule(data_config)
    train_loader, val_loader, test_loader = data_module.get_dataloaders(config.batch_size)

    # Initialize model
    model = DualMedicalSummarizer(
        model_name=config.model_name,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
        freeze_encoder=config.freeze_encoder
    )

    # Initialize trainer
    trainer = DualMedicalTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()