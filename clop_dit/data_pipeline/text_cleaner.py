# text_cleaner.py — SFT-based metadata text cleaning via Llama-3
"""
Text cleaning pipeline using fine-tuned Llama-3 to extract
structured biological metadata from messy GEO descriptions.

Pipeline:
    1. Load raw GEO metadata JSON
    2. Construct extraction prompt per sample
    3. Run through fine-tuned Llama-3 (via Unsloth)
    4. Parse structured JSON output
    5. Validate and save clean metadata

Structured output schema:
    {
        "tissue": str,
        "disease": str,
        "cell_type": str,
        "organism": str,
        "drug_treatment": str,
        "perturbation": str,
        "developmental_stage": str,
        "sequencing_platform": str,
        "condition_summary": str  # Natural language summary
    }
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================================
#  Extraction Prompt Template
# ============================================================================

EXTRACTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a biomedical metadata extraction assistant. Given raw GEO dataset metadata,
extract structured biological information into a JSON object.

Output ONLY valid JSON with these fields:
- tissue: anatomical tissue/organ (e.g., "lung", "brain", "liver")
- disease: disease/condition (e.g., "lung adenocarcinoma", "healthy")
- cell_type: specific cell type if mentioned (e.g., "T cell", "macrophage", "mixed")
- organism: species (e.g., "Homo sapiens", "Mus musculus")
- drug_treatment: drug/compound if mentioned (e.g., "cisplatin", "none")
- perturbation: genetic perturbation if any (e.g., "KRAS knockout", "none")
- developmental_stage: stage if mentioned (e.g., "E14.5", "adult")
- sequencing_platform: platform (e.g., "10x Chromium 3' v3", "Smart-seq2")
- condition_summary: one-sentence natural language description

If a field is not determinable, use "unknown".
<|eot_id|><|start_header_id|>user<|end_header_id|>
Extract biological metadata from this GEO record:

Title: {title}
Summary: {summary}
Overall Design: {overall_design}
Sample Source: {source}
Sample Characteristics: {characteristics}
Organism: {organism}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


class TextCleaner:
    """SFT-based text cleaning pipeline using Llama-3 via Unsloth.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path to fine-tuned adapter.
    adapter_path : str or None
        Path to LoRA adapter weights (if using fine-tuned version).
    max_seq_length : int
        Maximum sequence length for generation.
    device : str
        Target device.
    load_in_4bit : bool
        Whether to load in 4-bit quantization.
    """

    def __init__(
        self,
        model_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        adapter_path: Optional[str] = None,
        max_seq_length: int = 2048,
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the SFT model."""
        if self._model is not None:
            return

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is required for SFT text cleaning.\n"
                "Install via: pip install unsloth"
            )

        logger.info(f"Loading SFT model: {self.model_name}")

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto
            load_in_4bit=self.load_in_4bit,
        )

        # Load adapter if provided
        if self.adapter_path:
            logger.info(f"Loading LoRA adapter from {self.adapter_path}")
            self._model.load_adapter(self.adapter_path)

        FastLanguageModel.for_inference(self._model)
        logger.info("SFT model loaded and ready for inference.")

    def build_prompt(self, metadata: Dict) -> str:
        """Build extraction prompt from raw metadata.

        Parameters
        ----------
        metadata : dict
            Raw GEO metadata for a single sample/series.

        Returns
        -------
        prompt : str
        """
        return EXTRACTION_PROMPT.format(
            title=metadata.get("title", "unknown"),
            summary=metadata.get("summary", "unknown"),
            overall_design=metadata.get("overall_design", "unknown"),
            source=metadata.get("source", "unknown"),
            characteristics=str(metadata.get("characteristics", [])),
            organism=metadata.get("organism", "unknown"),
        )

    def extract_single(self, metadata: Dict) -> Dict:
        """Extract structured metadata from a single record.

        Parameters
        ----------
        metadata : dict
            Raw metadata.

        Returns
        -------
        structured : dict
            Parsed structured metadata.
        """
        self._load_model()

        prompt = self.build_prompt(metadata)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
        )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response
        try:
            # Find JSON block in response
            json_start = response.rfind("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                structured = json.loads(response[json_start:json_end])
            else:
                logger.warning("No JSON found in model output, returning defaults")
                structured = self._default_metadata()
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Using defaults.")
            structured = self._default_metadata()

        return structured

    def extract_batch(
        self,
        metadata_file: Union[str, Path],
        output_file: Union[str, Path],
    ) -> Dict:
        """Process all metadata entries and save structured output.

        Parameters
        ----------
        metadata_file : path
            Path to raw metadata JSON (from GEOFetcher).
        output_file : path
            Path to save structured metadata.

        Returns
        -------
        all_structured : dict
        """
        with open(metadata_file) as f:
            raw_metadata = json.load(f)

        all_structured = {}

        for gse_id, gse_meta in raw_metadata.items():
            logger.info(f"Processing {gse_id}...")

            # Series-level extraction
            series_struct = self.extract_single(gse_meta)
            all_structured[gse_id] = {
                "series": series_struct,
                "samples": {},
            }

            # Sample-level extraction
            for gsm_id, gsm_meta in gse_meta.get("samples", {}).items():
                sample_struct = self.extract_single(gsm_meta)
                all_structured[gse_id]["samples"][gsm_id] = sample_struct

        # Save
        with open(output_file, "w") as f:
            json.dump(all_structured, f, indent=2, ensure_ascii=False)

        logger.info(f"Structured metadata saved to {output_file}")
        return all_structured

    @staticmethod
    def _default_metadata() -> Dict:
        return {
            "tissue": "unknown",
            "disease": "unknown",
            "cell_type": "unknown",
            "organism": "unknown",
            "drug_treatment": "none",
            "perturbation": "none",
            "developmental_stage": "unknown",
            "sequencing_platform": "unknown",
            "condition_summary": "unknown biological sample",
        }

    def create_natural_text(self, structured: Dict) -> str:
        """Convert structured metadata to natural language for the BiomedBERT text encoder.

        Parameters
        ----------
        structured : dict
            Structured metadata from extract_single.

        Returns
        -------
        text : str
            Natural language description suitable for BERT encoding.
        """
        parts = []

        if structured.get("tissue", "unknown") != "unknown":
            parts.append(f"{structured['tissue']} tissue")
        if structured.get("disease", "unknown") != "unknown":
            parts.append(f"with {structured['disease']}")
        if structured.get("cell_type", "unknown") not in ("unknown", "mixed"):
            parts.append(f"focusing on {structured['cell_type']} cells")
        if structured.get("drug_treatment", "none") != "none":
            parts.append(f"treated with {structured['drug_treatment']}")
        if structured.get("perturbation", "none") != "none":
            parts.append(f"under {structured['perturbation']}")
        if structured.get("organism", "unknown") != "unknown":
            parts.append(f"from {structured['organism']}")

        if parts:
            return "Single-cell RNA sequencing of " + " ".join(parts) + "."
        else:
            return structured.get("condition_summary", "Single-cell RNA sequencing data.")


# ============================================================================
#  SFT Training Data Creator
# ============================================================================

class SFTDatasetBuilder:
    """Build training data for Llama-3 fine-tuning.

    Creates instruction-following pairs from manually annotated examples.

    Parameters
    ----------
    output_path : str or Path
        Path to save JSONL training file.
    """

    def __init__(self, output_path: Union[str, Path] = "data/sft_training.jsonl"):
        self.output_path = Path(output_path)
        self.examples = []

    def add_example(self, raw_metadata: Dict, structured_output: Dict):
        """Add a training example.

        Parameters
        ----------
        raw_metadata : dict
            Raw GEO metadata.
        structured_output : dict
            Manually annotated structured output.
        """
        prompt = EXTRACTION_PROMPT.format(
            title=raw_metadata.get("title", ""),
            summary=raw_metadata.get("summary", ""),
            overall_design=raw_metadata.get("overall_design", ""),
            source=raw_metadata.get("source", ""),
            characteristics=str(raw_metadata.get("characteristics", [])),
            organism=raw_metadata.get("organism", ""),
        )

        completion = json.dumps(structured_output, indent=2)

        self.examples.append({
            "instruction": prompt,
            "output": completion,
        })

    def save(self):
        """Save training data as JSONL."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.examples)} SFT training examples to {self.output_path}")
