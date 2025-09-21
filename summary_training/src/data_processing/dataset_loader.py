"""
Medical Dialog Dataset Loader and Processor
Handles loading, cleaning, and preprocessing of medical_dialog dataset.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationExample:
    """Structured representation of a medical conversation."""
    id: str
    turns: List[Dict[str, str]]  # [{"speaker": "patient/doctor", "text": "..."}]
    disease_tag: Optional[str] = None
    original_summary: Optional[str] = None
    phi_metadata: Optional[Dict] = None

class MedicalDialogLoader:
    """Load and preprocess medical_dialog dataset."""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, split: str = "train", language: str = "en") -> List[ConversationExample]:
        """
        Load medical_dialog dataset from HuggingFace.

        Args:
            split: train/validation/test
            language: en/zh (English/Chinese)
        """
        logger.info(f"Loading medical_dialog dataset: {split} split, {language} language")

        try:
            # Load from HuggingFace
            if language == "en":
                dataset = load_dataset("UCSD-DBMI/medical_dialog", "processed", split=split, cache_dir=str(self.cache_dir))
            else:
                dataset = load_dataset("UCSD-DBMI/medical_dialog", "zh", split=split, cache_dir=str(self.cache_dir))

            conversations = []

            for idx, example in enumerate(dataset):
                if self._is_valid_conversation(example):
                    conversation = self._parse_conversation(example, idx)
                    conversations.append(conversation)

            logger.info(f"Loaded {len(conversations)} valid conversations")
            return conversations

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            # Fallback to synthetic data for development
            return self._generate_synthetic_conversations()

    def _is_valid_conversation(self, example: Dict) -> bool:
        """Validate if conversation meets quality criteria."""
        conversation = example.get('conversation', [])

        # Minimum conversation requirements
        if len(conversation) < 3:
            return False

        # Must have both patient and doctor turns
        speakers = {turn.get('speaker', '') for turn in conversation}
        if not ('patient' in speakers and 'doctor' in speakers):
            return False

        # Text quality checks
        total_text_length = sum(len(turn.get('text', '')) for turn in conversation)
        if total_text_length < 50:  # Too short
            return False

        return True

    def _parse_conversation(self, example: Dict, idx: int) -> ConversationExample:
        """Parse raw conversation data into structured format."""
        conversation = example.get('conversation', [])

        # Normalize speaker labels
        normalized_turns = []
        for turn in conversation:
            speaker = turn.get('speaker', '').lower()
            if speaker in ['patient', 'user', 'p']:
                speaker = 'patient'
            elif speaker in ['doctor', 'physician', 'dr', 'd']:
                speaker = 'doctor'
            elif speaker in ['system', 'assistant']:
                speaker = 'system'
            else:
                speaker = 'unknown'

            normalized_turns.append({
                'speaker': speaker,
                'text': turn.get('text', '').strip()
            })

        return ConversationExample(
            id=example.get('id', f"synthetic_{idx}"),
            turns=normalized_turns,
            disease_tag=example.get('disease_tag'),
            original_summary=example.get('summary')
        )

    def _generate_synthetic_conversations(self) -> List[ConversationExample]:
        """Generate synthetic conversations for development/testing."""
        logger.warning("Generating synthetic data - use real medical_dialog dataset for production")

        synthetic_conversations = [
            {
                "id": "synthetic_001",
                "turns": [
                    {"speaker": "patient", "text": "I've been having headaches for the past week"},
                    {"speaker": "doctor", "text": "Can you describe the pain? Is it throbbing or constant?"},
                    {"speaker": "patient", "text": "It's a constant, dull ache mostly in my forehead"},
                    {"speaker": "doctor", "text": "Have you had any vision changes or nausea?"},
                    {"speaker": "patient", "text": "No vision changes, but I did feel nauseous yesterday"},
                    {"speaker": "doctor", "text": "Let's check your blood pressure and consider some tests"}
                ],
                "disease_tag": "headache",
                "original_summary": "Patient presents with week-long constant frontal headaches with associated nausea."
            },
            {
                "id": "synthetic_002",
                "turns": [
                    {"speaker": "patient", "text": "I have been coughing for 3 days and feel tired"},
                    {"speaker": "doctor", "text": "Is the cough dry or are you bringing up phlegm?"},
                    {"speaker": "patient", "text": "It's mostly dry, but sometimes yellowish phlegm"},
                    {"speaker": "doctor", "text": "Any fever or chest pain?"},
                    {"speaker": "patient", "text": "Low-grade fever yesterday, no chest pain"},
                    {"speaker": "doctor", "text": "This sounds like a respiratory infection. Let's do a chest examination"}
                ],
                "disease_tag": "respiratory_infection",
                "original_summary": "3-day history of productive cough with fatigue and low-grade fever."
            }
        ]

        conversations = []
        for i, conv_data in enumerate(synthetic_conversations):
            conversation = ConversationExample(
                id=conv_data["id"],
                turns=conv_data["turns"],
                disease_tag=conv_data["disease_tag"],
                original_summary=conv_data["original_summary"]
            )
            conversations.append(conversation)

        return conversations

class PHISafeProcessor:
    """Process conversations with PHI redaction for safe training."""

    def __init__(self, redaction_module_path: str = "backend.redact"):
        """Initialize with PHI redaction capability."""
        try:
            import sys
            sys.path.append(".")
            from backend.redact import redact_text, get_phi_analysis, assert_no_phi
            self.redact_text = redact_text
            self.get_phi_analysis = get_phi_analysis
            self.assert_no_phi = assert_no_phi
            self.phi_available = True
            logger.info("PHI redaction module loaded successfully")
        except ImportError:
            logger.warning("PHI redaction module not available - using placeholder")
            self.phi_available = False

    def process_conversation(self, conversation: ConversationExample) -> ConversationExample:
        """Apply PHI redaction to conversation."""
        if not self.phi_available:
            return conversation

        processed_turns = []
        phi_metadata = {"total_phi_items": 0, "phi_types": set(), "turns_with_phi": 0}

        for i, turn in enumerate(conversation.turns):
            # Analyze and redact PHI
            phi_analysis = self.get_phi_analysis(turn['text'])
            redacted_text, mapping = self.redact_text(turn['text'])

            # Verify redaction
            redaction_valid = self.assert_no_phi(redacted_text)
            if not redaction_valid:
                logger.warning(f"PHI redaction failed for conversation {conversation.id}, turn {i}")
                continue

            # Track PHI metadata
            if mapping:
                phi_metadata["turns_with_phi"] += 1
                phi_metadata["total_phi_items"] += len(mapping)
                phi_metadata["phi_types"].update(phi_analysis.keys())

            processed_turns.append({
                'speaker': turn['speaker'],
                'text_original': turn['text'],
                'text_redacted': redacted_text,
                'phi_detected': bool(mapping),
                'phi_types': list(phi_analysis.keys()) if phi_analysis else [],
                'turn_idx': i
            })

        # Convert phi_types set to list for JSON serialization
        phi_metadata["phi_types"] = list(phi_metadata["phi_types"])

        return ConversationExample(
            id=conversation.id,
            turns=processed_turns,
            disease_tag=conversation.disease_tag,
            original_summary=conversation.original_summary,
            phi_metadata=phi_metadata
        )

class ConversationSegmenter:
    """Segment long conversations for model training."""

    def __init__(self, max_context_length: int = 512, overlap_length: int = 50):
        self.max_context_length = max_context_length
        self.overlap_length = overlap_length

    def segment_conversation(self, conversation: ConversationExample) -> List[Dict]:
        """Split conversation into segments suitable for training."""
        segments = []

        # Simple segmentation by turn count and token length
        current_segment = []
        current_length = 0

        for turn in conversation.turns:
            # Estimate token length (rough approximation)
            text = turn.get('text_redacted', turn.get('text', ''))
            turn_length = len(text.split())

            if current_length + turn_length > self.max_context_length and current_segment:
                # Save current segment
                segments.append(self._create_segment(current_segment, conversation))

                # Start new segment with overlap
                overlap_turns = current_segment[-self.overlap_length:] if len(current_segment) > self.overlap_length else current_segment
                current_segment = overlap_turns + [turn]
                current_length = sum(len(t.get('text_redacted', t.get('text', '')).split()) for t in current_segment)
            else:
                current_segment.append(turn)
                current_length += turn_length

        # Add final segment
        if current_segment:
            segments.append(self._create_segment(current_segment, conversation))

        return segments

    def _create_segment(self, turns: List[Dict], conversation: ConversationExample) -> Dict:
        """Create a training segment from turns."""
        return {
            'conversation_id': conversation.id,
            'turns': turns,
            'start_turn': turns[0].get('turn_idx', 0),
            'end_turn': turns[-1].get('turn_idx', len(turns)-1),
            'total_turns': len(turns),
            'disease_tag': conversation.disease_tag,
            'phi_metadata': conversation.phi_metadata
        }

class TrainingDataBuilder:
    """Build training examples for dual summarization."""

    def __init__(self, tokenizer_name: str = "t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Special tokens for medical summarization
        special_tokens = [
            "[PATIENT_SUMMARY]", "[CLINICIAN_SUMMARY]",
            "[PATIENT_TURN]", "[DOCTOR_TURN]", "[SYSTEM_TURN]",
            "[PHI_REDACTED]", "[PROVENANCE]"
        ]

        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def build_training_examples(self, segments: List[Dict]) -> List[Dict]:
        """Convert conversation segments to training examples."""
        training_examples = []

        for segment in segments:
            # Build input text
            input_text = self._format_conversation_input(segment)

            # Generate target summaries
            clinician_summary = self._generate_clinician_summary(segment)
            patient_summary = self._generate_patient_summary(segment)

            # Tokenize
            input_tokens = self.tokenizer(
                input_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            clinician_tokens = self.tokenizer(
                clinician_summary,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            patient_tokens = self.tokenizer(
                patient_summary,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            training_examples.append({
                'input_ids': input_tokens['input_ids'].squeeze(),
                'attention_mask': input_tokens['attention_mask'].squeeze(),
                'clinician_target_ids': clinician_tokens['input_ids'].squeeze(),
                'patient_target_ids': patient_tokens['input_ids'].squeeze(),
                'conversation_id': segment['conversation_id'],
                'segment_info': {
                    'start_turn': segment['start_turn'],
                    'end_turn': segment['end_turn'],
                    'total_turns': segment['total_turns']
                }
            })

        return training_examples

    def _format_conversation_input(self, segment: Dict) -> str:
        """Format conversation segment as model input."""
        formatted_turns = []

        for turn in segment['turns']:
            speaker = turn['speaker'].upper()
            text = turn.get('text_redacted', turn.get('text', ''))
            formatted_turns.append(f"[{speaker}_TURN] {text}")

        input_text = " ".join(formatted_turns)
        return f"Summarize this medical conversation: {input_text}"

    def _generate_clinician_summary(self, segment: Dict) -> str:
        """Generate clinician-focused summary with structured medical format."""
        turns = segment['turns']
        patient_statements = [t for t in turns if t['speaker'] == 'patient']
        doctor_statements = [t for t in turns if t['speaker'] == 'doctor']

        # Analyze content for medical structure
        symptoms = []
        history = []
        assessments = []
        plans = []

        for stmt in patient_statements:
            text = stmt.get('text_redacted', stmt.get('text', ''))
            turn_idx = stmt.get('turn_idx', 0)

            # Simple keyword-based categorization (in practice, use NLP/ML)
            text_lower = text.lower()
            if any(word in text_lower for word in ['pain', 'hurt', 'ache', 'feel', 'symptom']):
                symptoms.append(f"• {text} [S{turn_idx}]")
            elif any(word in text_lower for word in ['day', 'week', 'month', 'started', 'began']):
                history.append(f"• {text} [S{turn_idx}]")
            else:
                symptoms.append(f"• {text} [S{turn_idx}]")

        for stmt in doctor_statements:
            text = stmt.get('text_redacted', stmt.get('text', ''))
            turn_idx = stmt.get('turn_idx', 0)
            text_lower = text.lower()

            if any(word in text_lower for word in ['recommend', 'prescribe', 'treatment', 'plan']):
                plans.append(f"• {text} [S{turn_idx}]")
            elif any(word in text_lower for word in ['diagnosis', 'assess', 'likely', 'condition']):
                assessments.append(f"• {text} [S{turn_idx}]")

        # Build structured clinical summary
        summary_parts = ["[CLINICIAN_SUMMARY]", "", "## CLINICAL SUMMARY", ""]

        if symptoms:
            summary_parts.extend(["### Chief Complaint & Symptoms:"] + symptoms + [""])

        if history:
            summary_parts.extend(["### History of Present Illness:"] + history + [""])

        if assessments:
            summary_parts.extend(["### Assessment:"] + assessments + [""])

        if plans:
            summary_parts.extend(["### Plan:"] + plans + [""])

        # Add disease tag if available
        if segment.get('disease_tag'):
            summary_parts.extend([f"### Diagnostic Category: {segment['disease_tag']}", ""])

        return "\n".join(summary_parts)

    def _generate_patient_summary(self, segment: Dict) -> str:
        """Generate patient-friendly summary using clear, accessible language."""
        turns = segment['turns']
        patient_statements = [t for t in turns if t['speaker'] == 'patient']
        doctor_statements = [t for t in turns if t['speaker'] == 'doctor']

        # Categorize for patient-friendly presentation
        what_you_told_us = []
        what_we_found = []
        next_steps = []
        important_notes = []

        # Process patient statements - what they shared
        for stmt in patient_statements:
            text = stmt.get('text_redacted', stmt.get('text', ''))
            turn_idx = stmt.get('turn_idx', 0)

            # Convert to patient-friendly language
            friendly_text = self._make_patient_friendly(text)
            what_you_told_us.append(f"• You mentioned: {friendly_text} [S{turn_idx}]")

        # Process doctor statements - what was assessed/planned
        for stmt in doctor_statements:
            text = stmt.get('text_redacted', stmt.get('text', ''))
            turn_idx = stmt.get('turn_idx', 0)
            text_lower = text.lower()

            friendly_text = self._make_patient_friendly(text)

            if any(word in text_lower for word in ['recommend', 'prescribe', 'treatment', 'plan', 'follow']):
                next_steps.append(f"• {friendly_text} [S{turn_idx}]")
            elif any(word in text_lower for word in ['test', 'exam', 'check', 'find']):
                what_we_found.append(f"• {friendly_text} [S{turn_idx}]")
            else:
                important_notes.append(f"• {friendly_text} [S{turn_idx}]")

        # Build patient-friendly summary
        summary_parts = ["[PATIENT_SUMMARY]", "", "## Your Medical Visit Summary", ""]

        if what_you_told_us:
            summary_parts.extend(["### What You Shared With Us:"] + what_you_told_us + [""])

        if what_we_found:
            summary_parts.extend(["### What We Discussed:"] + what_we_found + [""])

        if next_steps:
            summary_parts.extend(["### Your Next Steps:"] + next_steps + [""])

        if important_notes:
            summary_parts.extend(["### Important Things to Remember:"] + important_notes + [""])

        # Add simple condition explanation if available
        if segment.get('disease_tag'):
            condition = segment['disease_tag'].replace('_', ' ').title()
            summary_parts.extend([f"### About Your Condition:", f"We discussed concerns related to: **{condition}**", ""])

        summary_parts.extend([
            "### Questions or Concerns?",
            "• Please contact us if your symptoms get worse",
            "• Don't hesitate to call if you have questions about your treatment",
            "• Follow up as scheduled to monitor your progress",
            ""
        ])

        return "\n".join(summary_parts)

    def _make_patient_friendly(self, medical_text: str) -> str:
        """Convert medical terminology to patient-friendly language."""
        # Simple term replacements (in practice, use medical NLP)
        replacements = {
            'diagnosis': 'condition',
            'prescribe': 'give you medicine for',
            'medication': 'medicine',
            'symptoms': 'how you\'re feeling',
            'examination': 'check-up',
            'assessment': 'review',
            'hypertension': 'high blood pressure',
            'inflammation': 'swelling',
            'acute': 'sudden',
            'chronic': 'ongoing',
            'administer': 'give',
            'monitor': 'keep track of',
            'contraindicated': 'not safe for you',
            'adverse': 'unwanted side',
        }

        friendly_text = medical_text
        for medical_term, friendly_term in replacements.items():
            friendly_text = friendly_text.replace(medical_term, friendly_term)

        return friendly_text

class MedicalDialogDataModule:
    """Complete data processing pipeline."""

    def __init__(self, config: Dict):
        self.config = config
        self.loader = MedicalDialogLoader(config.get('cache_dir', './data/cache'))
        self.phi_processor = PHISafeProcessor()
        self.segmenter = ConversationSegmenter(
            max_context_length=config.get('max_context_length', 512)
        )
        self.builder = TrainingDataBuilder(config.get('tokenizer_name', 't5-base'))

    def prepare_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Prepare train/val/test datasets."""
        logger.info("Loading and processing medical dialog dataset...")

        # Load raw conversations
        conversations = self.loader.load_dataset(split="train")

        # Process with PHI redaction
        processed_conversations = []
        for conv in conversations:
            processed_conv = self.phi_processor.process_conversation(conv)
            processed_conversations.append(processed_conv)

        # Segment conversations
        all_segments = []
        for conv in processed_conversations:
            segments = self.segmenter.segment_conversation(conv)
            all_segments.extend(segments)

        # Build training examples
        training_examples = self.builder.build_training_examples(all_segments)

        # Split data
        np.random.shuffle(training_examples)
        n_train = int(len(training_examples) * 0.8)
        n_val = int(len(training_examples) * 0.1)

        train_data = training_examples[:n_train]
        val_data = training_examples[n_train:n_train+n_val]
        test_data = training_examples[n_train+n_val:]

        logger.info(f"Prepared {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")

        return train_data, val_data, test_data

    def get_dataloaders(self, batch_size: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get PyTorch DataLoaders."""
        train_data, val_data, test_data = self.prepare_data()

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

# Example usage and testing
if __name__ == "__main__":
    # Test the data processing pipeline
    config = {
        'cache_dir': './data/cache',
        'max_context_length': 512,
        'tokenizer_name': 't5-base'
    }

    data_module = MedicalDialogDataModule(config)

    # Test data loading
    train_data, val_data, test_data = data_module.prepare_data()

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    # Show example
    if train_data:
        example = train_data[0]
        print(f"\nExample training instance:")
        print(f"  Conversation ID: {example['conversation_id']}")
        print(f"  Input shape: {example['input_ids'].shape}")
        print(f"  Segment info: {example['segment_info']}")