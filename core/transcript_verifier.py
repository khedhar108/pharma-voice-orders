import json
import os
from typing import List, Dict, Union
from huggingface_hub import InferenceClient

class TranscriptVerifier:
    def __init__(self, hf_token: str = None, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        """
        Initialize the Transcript Verifier using LLM.
        
        Args:
            hf_token: HuggingFace API token.
            model_name: The LLM model to use (default: Zephyr 7B Beta).
        """
        self.token = hf_token or os.environ.get("HF_TOKEN")
        self.model_name = model_name
        
        if self.token:
            self.client = InferenceClient(token=self.token)
        else:
            print("Warning: No HF_TOKEN provided. TranscriptVerifier will fail if called.")
            self.client = None

    def verify_and_extract(self, transcript: str, possible_medicines: List[str]) -> Dict:
        """
        Clean transcript and extract structured entities using LLM.
        
        Args:
            transcript: Raw ASR transcript.
            possible_medicines: List of medicine names for context (fuzzy candidates).
            
        Returns:
            Dict containing 'cleaned_transcript' and 'entities' (list of dicts).
        """
        if not self.client:
            return {"error": "No HF Tool configured", "cleaned_transcript": transcript, "entities": []}

        # Contextual prompt for Zephyr-7b
        context_meds = ", ".join(possible_medicines[:30]) # Limit to avoid token overflow
        
        prompt = f"""<|system|>
You are an expert pharmacist AI. Your task is to:
1. Correct spelling mistakes in the medical transcript based on the context list.
2. Extract medicine orders into strict JSON format with fields: medicine, dosage, quantity, form.
3. Separate attached numbers (e.g. "Dolo650" -> "Dolo", "650mg").

Context Matches (Known Medicines): [{context_meds}]

Output Format (JSON only):
{{
  "cleaned_text": "Full corrected sentence",
  "entities": [
    {{ "medicine": "Name", "dosage": "Strength", "quantity": "Number", "form": "Type" }}
  ]
}}
</s>
<|user|>
Transcript: "{transcript}"
</s>
<|assistant|>"""

        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=512,
                temperature=0.1, # Low temp for precision
                return_full_text=False
            )
            
            # Parse JSON from response
            json_str = response.strip()
            # Attempt to find JSON block if wrapped
            if "{" in json_str:
                json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
                
            data = json.loads(json_str)
            return data
            
        except Exception as e:
            print(f"LLM Verification Error: {e}")
            return {"error": str(e), "cleaned_text": transcript, "entities": []}
