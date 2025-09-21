"""
Dual Medical Summarization Model
Generates both clinician and patient summaries with provenance tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    T5ForConditionalGeneration, T5Config, T5Tokenizer,
    BartForConditionalGeneration, BartConfig, BartTokenizer,
    AutoTokenizer, AutoModel
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ProvenanceAttention(nn.Module):
    """Attention mechanism for tracking source attribution in summaries."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Attention components
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Provenance-specific components
        self.segment_embedding = nn.Embedding(100, hidden_size)  # Max 100 segments
        self.provenance_scorer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, target_len, hidden_size]
            key: [batch_size, source_len, hidden_size]
            value: [batch_size, source_len, hidden_size]
            attention_mask: [batch_size, source_len]
            segment_ids: [batch_size, source_len] - segment ID for each token

        Returns:
            attended_output: [batch_size, target_len, hidden_size]
            provenance_scores: [batch_size, target_len, source_len]
        """
        batch_size, target_len, _ = query.shape
        source_len = key.shape[1]

        # Project to query, key, value
        Q = self.query_proj(query)  # [batch_size, target_len, hidden_size]
        K = self.key_proj(key)      # [batch_size, source_len, hidden_size]
        V = self.value_proj(value)  # [batch_size, source_len, hidden_size]

        # Add segment embeddings to keys if provided
        if segment_ids is not None:
            segment_embeds = self.segment_embedding(segment_ids)  # [batch_size, source_len, hidden_size]
            K = K + segment_embeds

        # Reshape for multi-head attention
        Q = Q.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        # [batch_size, num_heads, target_len, source_len]

        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, source_len]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch_size, num_heads, target_len, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, target_len, self.hidden_size)

        # Project output
        attended_output = self.out_proj(attended)

        # Compute provenance scores (average across heads)
        provenance_scores = attention_weights.mean(dim=1)  # [batch_size, target_len, source_len]

        return attended_output, provenance_scores

class TaskSpecificHead(nn.Module):
    """Task-specific head for clinician or patient summary generation."""

    def __init__(self, hidden_size: int, vocab_size: int, target_type: str):
        super().__init__()
        self.target_type = target_type
        self.hidden_size = hidden_size

        # Task-specific transformations
        self.task_embedding = nn.Embedding(2, hidden_size)  # 0: clinician, 1: patient
        self.style_adaptor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        # Initialize task ID
        self.task_id = 0 if target_type == "clinician" else 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Add task-specific embedding
        task_embed = self.task_embedding(torch.tensor(self.task_id, device=hidden_states.device))
        task_embed = task_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Apply task-specific style adaptation
        adapted_states = hidden_states + task_embed
        adapted_states = self.style_adaptor(adapted_states)

        # Project to vocabulary
        logits = self.output_projection(adapted_states)

        return logits

class DualMedicalSummarizer(nn.Module):
    """
    Dual Medical Summarization Model
    Generates both clinician and patient summaries with provenance tracking.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        max_source_length: int = 512,
        max_target_length: int = 256,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.model_name = model_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Load base model
        if "t5" in model_name.lower():
            self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model_type = "t5"
        elif "bart" in model_name.lower():
            self.base_model = BartForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model_type = "bart"
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        # Add medical-specific tokens
        special_tokens = [
            "[PATIENT_SUMMARY]", "[CLINICIAN_SUMMARY]",
            "[PATIENT_TURN]", "[DOCTOR_TURN]", "[SYSTEM_TURN]",
            "[PHI_REDACTED]", "[PROVENANCE]"
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Model configuration
        config = self.base_model.config
        self.hidden_size = config.d_model
        self.vocab_size = config.vocab_size

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.base_model.encoder.parameters():
                param.requires_grad = False

        # Task-specific components
        self.clinician_head = TaskSpecificHead(self.hidden_size, self.vocab_size, "clinician")
        self.patient_head = TaskSpecificHead(self.hidden_size, self.vocab_size, "patient")

        # Provenance attention
        self.provenance_attention = ProvenanceAttention(self.hidden_size)

        # Loss functions
        self.criterion = CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clinician_target_ids: Optional[torch.Tensor] = None,
        patient_target_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        target_type: str = "both"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for dual summarization.

        Args:
            input_ids: [batch_size, source_len]
            attention_mask: [batch_size, source_len]
            clinician_target_ids: [batch_size, target_len] (optional)
            patient_target_ids: [batch_size, target_len] (optional)
            segment_ids: [batch_size, source_len] (optional)
            target_type: "clinician", "patient", or "both"

        Returns:
            outputs: Dictionary containing logits, loss, and provenance scores
        """
        # Encode input
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        outputs = {}

        # Generate clinician summary
        if target_type in ["clinician", "both"]:
            clinician_outputs = self._generate_summary(
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                target_ids=clinician_target_ids,
                head=self.clinician_head,
                segment_ids=segment_ids,
                summary_type="clinician"
            )
            outputs.update({
                "clinician_logits": clinician_outputs["logits"],
                "clinician_loss": clinician_outputs["loss"],
                "clinician_provenance": clinician_outputs["provenance_scores"]
            })

        # Generate patient summary
        if target_type in ["patient", "both"]:
            patient_outputs = self._generate_summary(
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                target_ids=patient_target_ids,
                head=self.patient_head,
                segment_ids=segment_ids,
                summary_type="patient"
            )
            outputs.update({
                "patient_logits": patient_outputs["logits"],
                "patient_loss": patient_outputs["loss"],
                "patient_provenance": patient_outputs["provenance_scores"]
            })

        return outputs

    def _generate_summary(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor],
        head: TaskSpecificHead,
        segment_ids: Optional[torch.Tensor],
        summary_type: str
    ) -> Dict[str, torch.Tensor]:
        """Generate summary for specific target type."""
        batch_size = encoder_hidden_states.shape[0]

        if target_ids is not None:
            # Training mode - use teacher forcing
            if self.model_type == "t5":
                decoder_input_ids = self.base_model._shift_right(target_ids)
            else:  # BART
                decoder_input_ids = target_ids[:, :-1]
                target_ids = target_ids[:, 1:]

            # Decode with base model
            decoder_outputs = self.base_model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )

            decoder_hidden_states = decoder_outputs.last_hidden_state

            # Apply provenance attention
            attended_states, provenance_scores = self.provenance_attention(
                query=decoder_hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attention_mask=attention_mask,
                segment_ids=segment_ids
            )

            # Apply task-specific head
            logits = head(attended_states)

            # Compute loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        else:
            # Inference mode - generate step by step
            max_length = self.max_target_length
            device = encoder_hidden_states.device

            # Start with appropriate prefix token
            if summary_type == "clinician":
                start_token_id = self.tokenizer.convert_tokens_to_ids("[CLINICIAN_SUMMARY]")
            else:
                start_token_id = self.tokenizer.convert_tokens_to_ids("[PATIENT_SUMMARY]")

            decoder_input_ids = torch.tensor([[start_token_id]], device=device).expand(batch_size, 1)

            generated_tokens = []
            provenance_scores_list = []

            for step in range(max_length):
                # Decode current step
                decoder_outputs = self.base_model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    return_dict=True
                )

                decoder_hidden_states = decoder_outputs.last_hidden_state

                # Apply provenance attention
                attended_states, step_provenance = self.provenance_attention(
                    query=decoder_hidden_states[:, -1:],  # Only last position
                    key=encoder_hidden_states,
                    value=encoder_hidden_states,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids
                )

                # Apply task-specific head
                step_logits = head(attended_states)  # [batch_size, 1, vocab_size]

                # Sample next token
                next_token_logits = step_logits[:, -1, :]  # [batch_size, vocab_size]
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]

                # Append to sequence
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=-1)
                generated_tokens.append(next_token_ids)
                provenance_scores_list.append(step_provenance)

                # Check for EOS token
                if torch.all(next_token_ids == self.tokenizer.eos_token_id):
                    break

            # Combine outputs
            logits = torch.cat([head(decoder_hidden_states[:, i:i+1]) for i in range(decoder_hidden_states.shape[1])], dim=1)
            provenance_scores = torch.cat(provenance_scores_list, dim=1) if provenance_scores_list else None
            loss = torch.tensor(0.0, device=device)

        return {
            "logits": logits,
            "loss": loss,
            "provenance_scores": provenance_scores
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_type: str = "both",
        max_length: Optional[int] = None,
        segment_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Generate summaries for given input.

        Args:
            input_ids: [batch_size, source_len]
            attention_mask: [batch_size, source_len]
            target_type: "clinician", "patient", or "both"
            max_length: Maximum generation length
            segment_ids: [batch_size, source_len] (optional)

        Returns:
            Generated summaries and provenance information
        """
        self.eval()
        max_length = max_length or self.max_target_length

        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                target_type=target_type
            )

            results = {}

            if target_type in ["clinician", "both"]:
                clinician_logits = outputs["clinician_logits"]
                clinician_tokens = torch.argmax(clinician_logits, dim=-1)
                clinician_texts = self.tokenizer.batch_decode(clinician_tokens, skip_special_tokens=True)

                results["clinician_summaries"] = clinician_texts
                results["clinician_provenance"] = outputs["clinician_provenance"]

            if target_type in ["patient", "both"]:
                patient_logits = outputs["patient_logits"]
                patient_tokens = torch.argmax(patient_logits, dim=-1)
                patient_texts = self.tokenizer.batch_decode(patient_tokens, skip_special_tokens=True)

                results["patient_summaries"] = patient_texts
                results["patient_provenance"] = outputs["patient_provenance"]

            return results

    def compute_total_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        clinician_weight: float = 0.6,
        patient_weight: float = 0.4
    ) -> torch.Tensor:
        """Compute weighted total loss for multi-task learning."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if "clinician_loss" in outputs:
            total_loss += clinician_weight * outputs["clinician_loss"]

        if "patient_loss" in outputs:
            total_loss += patient_weight * outputs["patient_loss"]

        return total_loss

    def get_provenance_mapping(
        self,
        provenance_scores: torch.Tensor,
        input_tokens: List[str],
        summary_tokens: List[str],
        threshold: float = 0.1
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Extract provenance mapping from attention scores.

        Args:
            provenance_scores: [target_len, source_len]
            input_tokens: List of source tokens
            summary_tokens: List of summary tokens
            threshold: Minimum attention score threshold

        Returns:
            List of provenance mappings for each summary token
        """
        provenance_mapping = []

        for i, summary_token in enumerate(summary_tokens):
            if i < provenance_scores.shape[0]:
                attention_weights = provenance_scores[i]  # [source_len]

                # Find source tokens with attention above threshold
                significant_sources = []
                for j, weight in enumerate(attention_weights):
                    if weight > threshold and j < len(input_tokens):
                        significant_sources.append({
                            "token": input_tokens[j],
                            "weight": float(weight),
                            "position": j
                        })

                provenance_mapping.append({
                    "summary_token": summary_token,
                    "position": i,
                    "sources": significant_sources
                })

        return provenance_mapping

# Configuration classes
class DualSummarizerConfig:
    """Configuration for dual medical summarizer."""

    def __init__(
        self,
        model_name: str = "t5-base",
        max_source_length: int = 512,
        max_target_length: int = 256,
        freeze_encoder: bool = False,
        clinician_weight: float = 0.6,
        patient_weight: float = 0.4,
        provenance_weight: float = 0.2
    ):
        self.model_name = model_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.freeze_encoder = freeze_encoder
        self.clinician_weight = clinician_weight
        self.patient_weight = patient_weight
        self.provenance_weight = provenance_weight

# Example usage and testing
if __name__ == "__main__":
    # Test model initialization
    config = DualSummarizerConfig()
    model = DualMedicalSummarizer(
        model_name=config.model_name,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length
    )

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch_size = 2
    source_len = 256
    target_len = 128

    input_ids = torch.randint(0, model.vocab_size, (batch_size, source_len))
    attention_mask = torch.ones(batch_size, source_len)
    clinician_targets = torch.randint(0, model.vocab_size, (batch_size, target_len))
    patient_targets = torch.randint(0, model.vocab_size, (batch_size, target_len))

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        clinician_target_ids=clinician_targets,
        patient_target_ids=patient_targets,
        target_type="both"
    )

    print(f"Output keys: {list(outputs.keys())}")
    print(f"Clinician logits shape: {outputs['clinician_logits'].shape}")
    print(f"Patient logits shape: {outputs['patient_logits'].shape}")

    # Test generation
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_type="both"
    )

    print(f"Generated output keys: {list(generated.keys())}")
    if "clinician_summaries" in generated:
        print(f"Number of clinician summaries: {len(generated['clinician_summaries'])}")
    if "patient_summaries" in generated:
        print(f"Number of patient summaries: {len(generated['patient_summaries'])}")