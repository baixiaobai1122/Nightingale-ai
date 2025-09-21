"""
Medical Summarization Model Inference Interface
"""
import torch
from typing import List, Tuple
from transformers import AutoTokenizer
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class MedicalSummarizerInference:
    """Production inference interface for medical summarization."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the inference model."""
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")

        # Load the trained model (placeholder for now)
        self.model = None  # Will be loaded after training
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

        # Add special tokens
        special_tokens = [
            "[PATIENT_SUMMARY]", "[CLINICIAN_SUMMARY]",
            "[PATIENT_TURN]", "[DOCTOR_TURN]", "[SYSTEM_TURN]",
            "[PHI_REDACTED]", "[PROVENANCE]"
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def make_dual_summaries(self, spans: List[Tuple[int, str]]) -> Tuple[str, str]:
        """
        Generate dual summaries using the trained model.

        Args:
            spans: List of (span_idx, redacted_text) tuples

        Returns:
            Tuple of (clinician_summary, patient_summary)
        """
        if self.model is None:
            print("⚠️  Model not loaded yet. Using fallback template-based generation...")
            return self._fallback_generation(spans)

        # Format input for the model
        input_text = self._format_input(spans)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # Generate clinician summary
            clinician_outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                task_type="clinician"
            )

            # Generate patient summary
            patient_outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                task_type="patient"
            )

        # Decode outputs
        clinician_summary = self.tokenizer.decode(
            clinician_outputs[0],
            skip_special_tokens=True
        )

        patient_summary = self.tokenizer.decode(
            patient_outputs[0],
            skip_special_tokens=True
        )

        return clinician_summary, patient_summary

    def _format_input(self, spans: List[Tuple[int, str]]) -> str:
        """Format conversation spans for model input."""
        formatted_turns = []
        for span_idx, text in spans:
            # Simple speaker inference (in practice, this would be more sophisticated)
            if any(word in text.lower() for word in ['i', 'my', 'me', 'feel']):
                speaker = "PATIENT"
            else:
                speaker = "DOCTOR"
            formatted_turns.append(f"[{speaker}_TURN] {text}")

        input_text = " ".join(formatted_turns)
        return f"Summarize this medical conversation: {input_text}"

    def _fallback_generation(self, spans: List[Tuple[int, str]]) -> Tuple[str, str]:
        """Fallback to rule-based generation when model is not available."""
        from backend.summarize import make_dual_summaries
        return make_dual_summaries(spans)


# Global inference instance (singleton pattern)
_inference_model = None

def load_inference_model(model_path: str = None) -> MedicalSummarizerInference:
    """Load the global inference model."""
    global _inference_model

    if _inference_model is None:
        if model_path is None:
            model_path = "./outputs/medical_summarizer_v1/best_model"

        _inference_model = MedicalSummarizerInference(model_path)

    return _inference_model

def make_dual_summaries_ai(spans: List[Tuple[int, str]]) -> Tuple[str, str]:
    """
    AI-powered dual summary generation.
    This function will replace the rule-based backend.summarize.make_dual_summaries
    """
    model = load_inference_model()
    return model.make_dual_summaries(spans)


if __name__ == "__main__":
    # Test the inference interface
    test_spans = [
        (1, "I have been having chest pain for 2 days"),
        (2, "The pain gets worse when I exercise"),
        (3, "No previous heart problems"),
        (4, "Let's do an ECG and blood work"),
        (5, "Take it easy and follow up next week")
    ]

    print("=== Testing Medical Summarization Inference ===")

    model = load_inference_model()
    clinician_summary, patient_summary = model.make_dual_summaries(test_spans)

    print("\\n📋 Clinician Summary:")
    print(clinician_summary)

    print("\\n👤 Patient Summary:")
    print(patient_summary)