import re
import json
from typing import List, Dict
from pathlib import Path
from simulation.manufacturer_db import ManufacturerDB

class EntityExtractor:
    def __init__(self, db: ManufacturerDB):
        self.db = db
        self.aliases = self._load_aliases()
        
        # Form keywords that indicate a medicine nearby
        self.form_keywords = {
            'tablet': ['tablet', 'tab', 'tabs', 'capsule', 'cap', 'caps'],
            'syrup': ['syrup', 'liquid', 'suspension'],
            'injection': ['injection', 'inj', 'vial', 'ampoule'],
            'cream': ['cream', 'gel', 'ointment', 'tube'],
            'spray': ['spray', 'inhaler', 'puff'],
            'drops': ['drops', 'eye drops', 'ear drops'],
            'sachet': ['sachet', 'powder', 'granules']
        }
        
        # Unit keywords for quantity extraction
        self.unit_keywords = ['strips', 'strip', 'slips', 'slip', 'bottles', 'bottle', 
                              'tablets', 'tabs', 'pieces', 'pcs', 'boxes', 'box', 
                              'packs', 'pack', 'vials', 'vial', 'ampoules']
        
        # Spoken number mapping
        self.spoken_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
            'twenty-five': 25, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'two hundred': 200, 'three hundred': 300,
            'five hundred': 500, 'thousand': 1000
        }
        
    def _load_aliases(self) -> Dict:
        """Load pronunciation aliases from JSON file."""
        alias_path = Path("data/aliases.json")
        if alias_path.exists():
            with open(alias_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text for parsing."""
        text = text.lower()
        # Remove common ASR artifacts
        text = re.sub(r'</s>|<unk>|<s>', '', text)
        # Remove filler words
        text = re.sub(r'\b(uh|um|like|maybe|please|kindly)\b', '', text)
        # Normalize punctuation
        text = text.replace(",", " , ").replace(".", " ")
        # Convert spoken numbers to digits
        for word, num in self.spoken_numbers.items():
            text = re.sub(rf'\b{word}\b', str(num), text)
        return text.strip()
    
    def _resolve_alias(self, word: str) -> str:
        """Check if word is an alias for a known medicine."""
        word_lower = word.lower()
        for canonical, aliases in self.aliases.items():
            if word_lower in aliases or word_lower == canonical:
                return canonical
        return word
    
    def _extract_form(self, segment: str) -> str:
        """Extract form type from segment."""
        segment_lower = segment.lower()
        for form_type, keywords in self.form_keywords.items():
            for kw in keywords:
                if kw in segment_lower:
                    return form_type
        return "tablet"  # Default
    
    def _extract_quantity(self, segment: str) -> tuple:
        """Extract quantity and unit from segment."""
        # Pattern 1: Number followed by unit word
        # e.g., "300 strips", "20 bottles"
        qty_pattern = r'(\d+)\s*(' + '|'.join(self.unit_keywords) + r')?'
        match = re.search(qty_pattern, segment, re.IGNORECASE)
        
        if match:
            num = match.group(1)
            unit = match.group(2) if match.group(2) else "units"
            # Normalize common typos
            if unit in ['slips', 'slip']:
                unit = 'strips'
            return num, unit
        
        return "1", "units"  # Default
    
    def _extract_dosage(self, segment: str) -> str:
        """Extract dosage from segment."""
        # Pattern: Number followed by mg/ml/gm
        dosage_match = re.search(r'(\d+)\s*(mg|ml|gm|mcg)', segment, re.IGNORECASE)
        if dosage_match:
            return f"{dosage_match.group(1)}{dosage_match.group(2)}"
        return "-"
        
    def extract(self, text: str) -> List[Dict]:
        """
        Extract medicine entities from text.
        Returns: List of dicts {'medicine': str, 'form': str, 'quantity': str, 'dosage': str}
        """
        if not text:
            return []
            
        # Normalize text
        text = self._normalize_text(text)
        
        found_orders = []
        
        # Get all known medicines from DB for matching
        known_meds = self.db.medicines['medicine_name'].tolist()
        
        # Split by multiple delimiters for multi-item orders
        # Handles: "send", "order", "add", "also", "plus", "then", "and", comma
        delimiters = r'\b(?:send|add|want|need|order|also|plus|then)\b|,|\band\b'
        segments = re.split(delimiters, text)
        
        for segment in segments:
            segment = segment.strip()
            if not segment or len(segment) < 3:
                continue
            
            # Try to find a medicine match in this segment
            from rapidfuzz import process, fuzz
            
            # First, check if any word is a known alias
            words = segment.split()
            resolved_segment = ' '.join([self._resolve_alias(w) for w in words])
            
            # Fuzzy match against known medicines
            match = process.extractOne(resolved_segment, known_meds, scorer=fuzz.partial_ratio)
            
            if match and match[1] > 75:  # Confidence threshold
                med_name = match[0]
                
                # Extract form, quantity, dosage
                form = self._extract_form(segment)
                num, unit = self._extract_quantity(segment)
                quantity = f"{num} {unit}"
                
                dosage = self._extract_dosage(segment)
                if dosage == "-":
                    # Lookup default dosage from DB
                    med_row = self.db.medicines[self.db.medicines['medicine_name'] == med_name].iloc[0]
                    dosage = med_row['dosage']
                
                found_orders.append({
                    "medicine": med_name,
                    "form": form,
                    "quantity": quantity,
                    "dosage": dosage,
                    "confidence": match[1],
                    "original_segment": segment.strip()
                })
        
        return found_orders
