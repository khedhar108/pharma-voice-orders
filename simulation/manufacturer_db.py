import json
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


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
        """Load medicines from the expanded dataset."""
        path = self.data_dir / "Medicine_Names_Packges.csv"
        
        # Fallback to small dataset if large one missing
        if not path.exists():
            path = self.data_dir / "medicines.csv"
            if not path.exists():
                return pd.DataFrame(columns=["medicine_name", "dosage", "unit", "manufacturer_id"])
            return pd.read_csv(path)

        try:
            # Load large dataset
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
            
        # Clean and map columns
        # ITEM NAME -> medicine_name
        # Full_Name_of_Manufacturer -> manufacturer_id (we will store Name here and map later)
        
        df = df.rename(columns={
            "ITEM NAME": "medicine_name",
            "Full_Name_of_Manufacturer": "manufacturer",
            "PACKING": "unit"
        })
        
        # Select relevant columns and drop NA
        df = df[['medicine_name', 'manufacturer', 'unit']].dropna(subset=['medicine_name'])
        
        # Ensure strings
        df['medicine_name'] = df['medicine_name'].astype(str).str.strip()
        df['manufacturer'] = df['manufacturer'].astype(str).str.strip()
        
        # Basic parsing of dosage from name (e.g., "DOLO-650" -> 650)
        # This is a simple heuristic; downstream LLM is better.
        df['dosage'] = "" 
        
        # Normalize manufacturer names to IDs if possible, else keep name
        # We need a way to map "Sun Pharma Laboratories Ltd" -> "mfr_001"
        # For now, we will dynamically create IDs or valid manufacturer names
        
        return df
    
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

    # Hardcoded mapping for prominent medicines to ensuring accuracy
    PROMINENT_MAPPING = {
        # ===== GENERIC NAMES (SALT NAMES) =====
        # Map generics to their most common brand manufacturer
        "paracetamol": "Micro Labs Ltd",  # Generic for Dolo, Calpol
        "acetaminophen": "Micro Labs Ltd",  # US name for paracetamol
        "azithromycin": "Alembic Pharmaceuticals",  # Generic for Azithral
        "amoxicillin": "GSK",  # Generic for Amoxil/Augmentin base
        "omeprazole": "Dr. Reddy's",  # Generic PPI
        "pantoprazole": "Alkem Laboratories",  # Generic for Pan-D
        "metformin": "USV Ltd",  # Generic for Glycomet
        "atorvastatin": "Zydus Cadila",  # Generic statin
        "cetirizine": "Cipla",  # Generic for Okacet
        "ibuprofen": "Abbott",  # Generic NSAID
        "aspirin": "USV Ltd",  # Ecosprin
        "ciprofloxacin": "Cipla",  # Generic for Ciplox
        "levofloxacin": "Cipla",  # Generic quinolone
        "cefixime": "FDC Ltd",  # Generic for Zifi
        "ranitidine": "JB Chemicals",  # Generic for Rantac
        "montelukast": "Cipla",  # Generic for Montair
        "levothyroxine": "Abbott",  # Generic for Thyronorm
        "amlodipine": "Dr. Reddy's",  # Generic for Stamlo/Amlong
        "telmisartan": "Glenmark",  # Generic for Telma
        "losartan": "Zydus Cadila",  # Generic ARB
        "metoprolol": "Cipla",  # Generic beta blocker
        
        # ===== ANALGESICS / ANTIPYRETICS =====
        "dolo": "Micro Labs Ltd",
        "dolo 650": "Micro Labs Ltd",
        "dolo-650": "Micro Labs Ltd",
        "calpol": "GSK",
        "paracip": "Cipla",
        "crocin": "GSK",
        "crocin advance": "GSK",
        "combiflam": "Sanofi India",
        "meftal": "Blue Cross",
        "meftal spas": "Blue Cross",
        "saridon": "Piramal",
        "disprin": "Reckitt Benckiser",
        "brufen": "Abbott",
        "ibugesic": "Cipla",
        
        # ===== ANTIBIOTICS =====
        "azithral": "Alembic Pharmaceuticals",
        "azithral 500": "Alembic Pharmaceuticals",
        "augmentin": "GSK",
        "augmentin 625": "GSK",
        "taxim": "Alkem Laboratories",
        "taxim o": "Alkem Laboratories",
        "zifi": "FDC Ltd",
        "zifi 200": "FDC Ltd",
        "monocef": "Aristo Pharmaceuticals",
        "ciplox": "Cipla",
        "ciplox 500": "Cipla",
        "moxikind": "Mankind Pharma",
        "moxikind cv": "Mankind Pharma",
        "clamvam": "Alkem Laboratories",
        "amoxyclav": "Alkem Laboratories",
        
        # ===== ACIDITY / GASTRIC =====
        "pan d": "Alkem Laboratories",
        "pan 40": "Alkem Laboratories",
        "pantop": "Aristo Pharmaceuticals",
        "pantop d": "Aristo Pharmaceuticals",
        "rantac": "JB Chemicals",
        "aciloc": "Cadila",
        "digene": "Abbott",
        "gelusil": "Pfizer",
        "omee": "Alkem Laboratories",
        "rabekind": "Mankind Pharma",
        "rablet": "Lupin",
        "pantocid": "Sun Pharma",
        "nexpro": "Torrent Pharmaceuticals",
        
        # ===== COLD / COUGH / ALLERGY =====
        "allegra": "Sanofi",
        "allegra 120": "Sanofi",
        "ascoril": "Glenmark",
        "ascoril d": "Glenmark",
        "benadryl": "Johnson & Johnson",
        "cofsils": "Cipla",
        "wikoryl": "Alembic Pharmaceuticals",
        "okacet": "Cipla",
        "levocet": "Hetero Healthcare",
        "montair": "Cipla",
        "montair lc": "Cipla",
        "cheston": "Cipla",
        "cheston cold": "Cipla",
        "grilinctus": "Franco-Indian",
        "sinarest": "Centaur Pharmaceuticals",
        "vicks action 500": "P&G",
        
        # ===== VITAMINS / SUPPLEMENTS =====
        "becosules": "Pfizer",
        "limcee": "Abbott",
        "shelcal": "Torrent Pharmaceuticals",
        "neurobion": "Procter & Gamble",
        "polybion": "Procter & Gamble",
        "revital": "Sun Pharma",
        "zincovit": "Apex Laboratories",
        "supradyn": "Bayer",
        "evion": "Merck",
        "calcium sandoz": "Novartis",
        
        # ===== CHRONIC (BP/SUGAR/THYROID) =====
        "telma": "Glenmark",
        "telma 40": "Glenmark",
        "glycomet": "USV Ltd",
        "glycomet 500": "USV Ltd",
        "thyronorm": "Abbott",
        "thyronorm 50": "Abbott",
        "amlong": "Micro Labs",
        "amlong 5": "Micro Labs",
        "stamlo": "Dr. Reddy's",
        "stamlo 5": "Dr. Reddy's",
        "human mixtard": "Novo Nordisk",
        "ecosprin": "USV Ltd",
        "ecosprin 75": "USV Ltd",
        "atorva": "Zydus Cadila",
        "rosulip": "USV Ltd",
        "cardace": "Aventis",
        "concor": "Merck",
        
        # ===== PAIN / MUSCLE RELAXANTS =====
        "flexon": "Aristo Pharmaceuticals",
        "flexon mr": "Aristo Pharmaceuticals",
        "zerodol": "Ipca Laboratories",
        "zerodol sp": "Ipca Laboratories",
        "hifenac": "Intas Pharmaceuticals",
        "hifenac p": "Intas Pharmaceuticals",
        "voveran": "Novartis",
        "voveran sr": "Novartis",
        "nicip": "Cipla",
        "nicip plus": "Cipla",
        
        # ===== OTHERS =====
        "betadine": "Win-Medicare",
        "liv 52": "Himalaya Wellness",
        "volini": "Sun Pharma",
        "move": "Reckitt Benckiser",
        "citralka": "Pfizer",
        "unwanted 72": "Mankind Pharma",
        "prega news": "Mankind Pharma",
        "manforce": "Mankind Pharma",
        "metrogyl": "JB Chemicals",
        "flagyl": "Sanofi",
        "norflox": "Cipla",
        "oflox": "Cipla"
    }

    def get_all_manufacturers(self) -> list:
        """Return list of manufacturer dicts, including dynamic ones."""
        mfrs = self.manufacturers.to_dict('records')
        # Ensure 'NewMed Technologies' is present for UI display
        if not any(m['name'] == 'NewMed Technologies' for m in mfrs):
            mfrs.append({
                "id": "mfr_newmed",
                "name": "NewMed Technologies",
                "code": "NMT"
            })
        return mfrs

    def get_manufacturer_by_medicine(self, medicine_name: str) -> dict:
        """
        Find manufacturer for a given medicine name.
        Strategy:
        1. Check Prominent Mapping (Exact/Lower match)
        2. Fuzzy Match against Database
        3. Fallback to 'NewMed Technologies' if confidence low
        """
        med_lower = medicine_name.lower().strip()
        
        # 1. Prominent Mapping
        for key, mfr_name in self.PROMINENT_MAPPING.items():
            if key in med_lower: # Simple substring check for prominence
                return {
                    "id": "mfr_prominent", 
                    "name": mfr_name,
                    "medicine_match": medicine_name, # Keep original name
                    "confidence": 100
                }

        # Resolve potential alias
        resolved_name = self._resolve_alias(medicine_name)
        
        # 2. Fuzzy Match
        if not self.medicines.empty:
            known_meds = self.medicines['medicine_name'].tolist()
            
            # Optimization: distinct meds only to speed up
            unique_meds = list(set(known_meds))
            
            # RapidFuzz extraction
            match = fuzz.process.extractOne(
                resolved_name, 
                unique_meds, 
                scorer=fuzz.token_set_ratio,
                score_cutoff=75
            )

            if match:
                dataset_med_name, confidence, _ = match
                
                # Get manufacturer from DB
                med_row = self.medicines[self.medicines['medicine_name'] == dataset_med_name].iloc[0]
                mfr_name = med_row.get('manufacturer', 'Unknown Pharma')
                
                # Construct ID if missing (simple hash or lookup)
                # We prioritize the name now
                
                return {
                    "id": "mfr_db_match",
                    "name": mfr_name,
                    "medicine_match": dataset_med_name,
                    "confidence": confidence
                }

        # 3. NewMed Fallback (The "Unlinked" Logic)
        return {
            "id": "mfr_newmed",
            "name": "NewMed Technologies",
            "medicine_match": medicine_name,
            "confidence": 0
        }

    def get_orders_by_manufacturer(self, current_orders: list) -> dict:
        """
        Group a list of extracted orders by manufacturer.
        Returns: { "Sun Pharma": [orders...], "Cipla": [orders...] }
        """
        # Initialize with known manufacturers to ensure they appear in UI output
        grouped = {mfr: [] for mfr in self.manufacturers['name'].tolist()}
        grouped['NewMed Technologies'] = [] # Ensure this key exists
        grouped['Unknown'] = []  # Fallback
        
        for order in current_orders:
            med_name = order.get('medicine')
            mfr_info = self.get_manufacturer_by_medicine(med_name)
            
            if mfr_info:
                mfr_name = mfr_info['name']
                # Update order with standardized name
                order['medicine_standardized'] = mfr_info['medicine_match']
                
                # Dynamically create key if it doesn't exist (e.g. from new prominent mapping)
                if mfr_name not in grouped:
                    grouped[mfr_name] = []
                    
                grouped[mfr_name].append(order)
            else:
                grouped['Unknown'].append(order)
                
        return grouped

