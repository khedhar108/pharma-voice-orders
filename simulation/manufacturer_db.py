import pandas as pd
import json
from pathlib import Path
from rapidfuzz import process, fuzz

class ManufacturerDB:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.manufacturers = self._load_manufacturers()
        self.medicines = self._load_medicines()
        self.aliases = self._load_aliases()

    def _load_manufacturers(self) -> pd.DataFrame:
        path = self.data_dir / "manufacturers.csv"
        if not path.exists():
            return pd.DataFrame(columns=["id", "name", "code"])
        return pd.read_csv(path)

    def _load_medicines(self) -> pd.DataFrame:
        path = self.data_dir / "medicines.csv"
        if not path.exists():
            return pd.DataFrame(columns=["medicine_name", "dosage", "unit", "manufacturer_id"])
        return pd.read_csv(path)
    
    def _load_aliases(self) -> dict:
        """Load pronunciation aliases from JSON file."""
        path = self.data_dir / "aliases.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def _resolve_alias(self, name: str) -> str:
        """Check if name is an alias for a known medicine."""
        name_lower = name.lower()
        for canonical, aliases in self.aliases.items():
            if name_lower in aliases or name_lower == canonical:
                return canonical
        return name

    def get_all_manufacturers(self) -> list:
        """Return list of manufacturer dicts."""
        return self.manufacturers.to_dict('records')

    def get_manufacturer_by_medicine(self, medicine_name: str) -> dict:
        """
        Find manufacturer for a given medicine name using fuzzy matching.
        Returns manufacturer dict or None.
        """
        # Resolve potential alias first
        resolved_name = self._resolve_alias(medicine_name)
        
        # Get list of known medicines
        known_meds = self.medicines['medicine_name'].tolist()
        
        # Composite Weighted Scorer (ULTRATHINK Strategy)
        # Prioritize structure (token_set) over pure substring (partial)
        # Weight: 60% Token Set (handles reordering "Dolo 650" == "650 Dolo")
        #         40% Partial Ratio (handles substrings "Dolo" in "Dolo 650")
        
        matches = []
        for candidate in known_meds:
            token_score = fuzz.token_set_ratio(resolved_name, candidate)
            partial_score = fuzz.partial_ratio(resolved_name, candidate)
            
            # Weighted Composite Score
            final_score = (0.60 * token_score) + (0.40 * partial_score)
            
            # Optimization: Only keep good matches to reduce sort time
            if final_score >= 70:  # Pre-filter
                matches.append((candidate, final_score))
        
        if not matches:
            return None
            
        # Get best match
        best_match = max(matches, key=lambda x: x[1])
        dataset_med_name, confidence = best_match
        
        # "Trust Cliff" - Safety Threshold
        # Using 75 as strict safety gate
        if confidence < 75:
            return None
            
        match = (dataset_med_name, confidence) # Compatible format
            
        dataset_med_name = match[0]
        
        # Look up manufacturer ID
        med_row = self.medicines[self.medicines['medicine_name'] == dataset_med_name].iloc[0]
        mfr_id = med_row['manufacturer_id']
        
        # Get manufacturer details
        mfr_row = self.manufacturers[self.manufacturers['id'] == mfr_id].iloc[0]
        
        return {
            "id": mfr_id,
            "name": mfr_row["name"],
            "medicine_match": dataset_med_name,  # Return the standardized name
            "confidence": match[1]
        }

    def get_orders_by_manufacturer(self, current_orders: list) -> dict:
        """
        Group a list of extracted orders by manufacturer.
        Returns: { "Sun Pharma": [orders...], "Cipla": [orders...] }
        """
        grouped = {mfr: [] for mfr in self.manufacturers['name'].tolist()}
        grouped['Unknown'] = []  # For unmapped medicines
        
        for order in current_orders:
            med_name = order.get('medicine')
            mfr_info = self.get_manufacturer_by_medicine(med_name)
            
            if mfr_info:
                # Update order with standardized name
                order['medicine_standardized'] = mfr_info['medicine_match']
                grouped[mfr_info['name']].append(order)
            else:
                grouped['Unknown'].append(order)
                
        return grouped

