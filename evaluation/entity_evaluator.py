"""
Entity Extraction Evaluator Module

Computes precision, recall, F1 score, and confusion matrix
for entity extraction by comparing against ground truth.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from rapidfuzz import fuzz
from dataclasses import dataclass, field


@dataclass
class MatchResult:
    """Result of matching expected vs extracted entity."""
    expected: Dict
    extracted: Dict = None
    match_type: str = "FN"  # TP, FP, FN
    medicine_match: bool = False
    quantity_match: bool = False
    dosage_match: bool = False


@dataclass
class EvaluationReport:
    """Complete evaluation report with metrics."""
    total_expected: int = 0
    total_extracted: int = 0
    
    # Per-field counts
    medicine_tp: int = 0
    medicine_fp: int = 0
    medicine_fn: int = 0
    
    quantity_tp: int = 0
    quantity_fp: int = 0
    quantity_fn: int = 0
    
    dosage_tp: int = 0
    dosage_fp: int = 0
    dosage_fn: int = 0
    
    # Detailed results
    matches: List[MatchResult] = field(default_factory=list)
    
    def precision(self, field: str = "medicine") -> float:
        tp = getattr(self, f"{field}_tp")
        fp = getattr(self, f"{field}_fp")
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def recall(self, field: str = "medicine") -> float:
        tp = getattr(self, f"{field}_tp")
        fn = getattr(self, f"{field}_fn")
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def f1_score(self, field: str = "medicine") -> float:
        p = self.precision(field)
        r = self.recall(field)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_expected": self.total_expected,
            "total_extracted": self.total_extracted,
            "medicine": {
                "tp": self.medicine_tp,
                "fp": self.medicine_fp,
                "fn": self.medicine_fn,
                "precision": round(self.precision("medicine") * 100, 1),
                "recall": round(self.recall("medicine") * 100, 1),
                "f1": round(self.f1_score("medicine") * 100, 1),
            },
            "quantity": {
                "tp": self.quantity_tp,
                "fp": self.quantity_fp,
                "fn": self.quantity_fn,
                "precision": round(self.precision("quantity") * 100, 1),
                "recall": round(self.recall("quantity") * 100, 1),
                "f1": round(self.f1_score("quantity") * 100, 1),
            },
            "dosage": {
                "tp": self.dosage_tp,
                "fp": self.dosage_fp,
                "fn": self.dosage_fn,
                "precision": round(self.precision("dosage") * 100, 1),
                "recall": round(self.recall("dosage") * 100, 1),
                "f1": round(self.f1_score("dosage") * 100, 1),
            },
        }


class EntityEvaluator:
    """
    Evaluates entity extraction accuracy against ground truth.
    
    Usage:
        evaluator = EntityEvaluator()
        report = evaluator.evaluate(ground_truth_df, extracted_results)
    """
    
    def __init__(self, medicine_threshold: int = 85, quantity_threshold: int = 80):
        self.medicine_threshold = medicine_threshold
        self.quantity_threshold = quantity_threshold
    
    def load_ground_truth(self, csv_path: str = "evaluation/ground_truth.csv") -> pd.DataFrame:
        """Load ground truth from CSV file."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {csv_path}")
        
        df = pd.read_csv(csv_path, comment='#')
        required_cols = ['audio_file', 'medicine_name', 'quantity', 'dosage']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def normalize_medicine(self, name: str) -> str:
        """Normalize medicine name for matching."""
        if not name:
            return ""
        return name.lower().strip().replace("-", " ").replace("_", " ")
    
    def normalize_quantity(self, qty: str) -> Tuple[int, str]:
        """Extract number and unit from quantity string."""
        if not qty or qty == "-":
            return 0, ""
        
        import re
        match = re.search(r'(\d+)\s*(\w+)?', str(qty).lower())
        if match:
            num = int(match.group(1))
            unit = match.group(2) or "units"
            # Normalize unit variations
            if unit in ['strip', 'strips', 'slip', 'slips']:
                unit = 'strips'
            elif unit in ['bottle', 'bottles']:
                unit = 'bottles'
            elif unit in ['pack', 'packs', 'packet', 'packets']:
                unit = 'packs'
            elif unit in ['box', 'boxes']:
                unit = 'boxes'
            return num, unit
        return 0, ""
    
    def normalize_dosage(self, dosage: str) -> str:
        """Normalize dosage string."""
        if not dosage or dosage == "-":
            return ""
        return str(dosage).lower().replace(" ", "").replace("-", "")
    
    def match_medicine(self, expected: str, extracted: str) -> bool:
        """Check if medicine names match (fuzzy)."""
        exp_norm = self.normalize_medicine(expected)
        ext_norm = self.normalize_medicine(extracted)
        
        if not exp_norm or not ext_norm:
            return False
        
        # Try exact match first
        if exp_norm == ext_norm:
            return True
        
        # Fuzzy match
        ratio = fuzz.ratio(exp_norm, ext_norm)
        partial_ratio = fuzz.partial_ratio(exp_norm, ext_norm)
        
        return max(ratio, partial_ratio) >= self.medicine_threshold
    
    def match_quantity(self, expected: str, extracted: str) -> bool:
        """Check if quantities match."""
        exp_num, exp_unit = self.normalize_quantity(expected)
        ext_num, ext_unit = self.normalize_quantity(extracted)
        
        if exp_num == 0:
            return True  # No expected quantity, skip check
        
        # Number must match exactly
        if exp_num != ext_num:
            return False
        
        # Unit should be similar
        if exp_unit and ext_unit:
            return fuzz.ratio(exp_unit, ext_unit) >= self.quantity_threshold
        
        return True
    
    def match_dosage(self, expected: str, extracted: str) -> bool:
        """Check if dosages match."""
        exp_norm = self.normalize_dosage(expected)
        ext_norm = self.normalize_dosage(extracted)
        
        if not exp_norm:
            return True  # No expected dosage, skip check
        
        return exp_norm == ext_norm
    
    def evaluate(
        self, 
        ground_truth: pd.DataFrame, 
        extracted_results: List[Dict],
        audio_file: str = None
    ) -> EvaluationReport:
        """
        Compare extracted results against ground truth.
        
        Args:
            ground_truth: DataFrame with expected entities
            extracted_results: List of dicts with extracted entities
            audio_file: Optional filter for specific audio file
            
        Returns:
            EvaluationReport with metrics
        """
        report = EvaluationReport()
        
        # Filter ground truth if audio_file specified
        if audio_file:
            gt = ground_truth[ground_truth['audio_file'] == audio_file].copy()
        else:
            gt = ground_truth.copy()
        
        expected_list = gt.to_dict('records')
        extracted_list = list(extracted_results)
        
        report.total_expected = len(expected_list)
        report.total_extracted = len(extracted_list)
        
        # Track which extracted items have been matched
        extracted_matched = [False] * len(extracted_list)
        
        # Match each expected item
        for exp in expected_list:
            exp_medicine = str(exp.get('medicine_name', ''))
            exp_quantity = str(exp.get('quantity', ''))
            exp_dosage = str(exp.get('dosage', ''))
            
            best_match = None
            best_match_idx = -1
            best_score = 0
            
            # Find best matching extracted item
            for i, ext in enumerate(extracted_list):
                if extracted_matched[i]:
                    continue
                
                ext_medicine = str(ext.get('medicine', ''))
                
                if self.match_medicine(exp_medicine, ext_medicine):
                    score = fuzz.ratio(
                        self.normalize_medicine(exp_medicine),
                        self.normalize_medicine(ext_medicine)
                    )
                    if score > best_score:
                        best_score = score
                        best_match = ext
                        best_match_idx = i
            
            if best_match:
                extracted_matched[best_match_idx] = True
                
                ext_quantity = str(best_match.get('quantity', ''))
                ext_dosage = str(best_match.get('dosage', ''))
                
                medicine_ok = True  # Already matched
                quantity_ok = self.match_quantity(exp_quantity, ext_quantity)
                dosage_ok = self.match_dosage(exp_dosage, ext_dosage)
                
                # Count TPs
                report.medicine_tp += 1
                if quantity_ok:
                    report.quantity_tp += 1
                else:
                    report.quantity_fn += 1
                if dosage_ok:
                    report.dosage_tp += 1
                else:
                    report.dosage_fn += 1
                
                report.matches.append(MatchResult(
                    expected=exp,
                    extracted=best_match,
                    match_type="TP",
                    medicine_match=True,
                    quantity_match=quantity_ok,
                    dosage_match=dosage_ok
                ))
            else:
                # False Negative - expected but not found
                report.medicine_fn += 1
                report.quantity_fn += 1
                report.dosage_fn += 1
                
                report.matches.append(MatchResult(
                    expected=exp,
                    extracted=None,
                    match_type="FN",
                    medicine_match=False,
                    quantity_match=False,
                    dosage_match=False
                ))
        
        # Count False Positives (extracted but not in ground truth)
        for i, matched in enumerate(extracted_matched):
            if not matched:
                report.medicine_fp += 1
                report.quantity_fp += 1
                report.dosage_fp += 1
                
                report.matches.append(MatchResult(
                    expected={},
                    extracted=extracted_list[i],
                    match_type="FP",
                    medicine_match=False,
                    quantity_match=False,
                    dosage_match=False
                ))
        
        return report
    
    def generate_comparison_table(self, report: EvaluationReport) -> pd.DataFrame:
        """Generate a comparison table from evaluation report."""
        rows = []
        
        for match in report.matches:
            exp = match.expected
            ext = match.extracted or {}
            
            rows.append({
                "audio_file": exp.get('audio_file', '-'),
                "expected_medicine": exp.get('medicine_name', '-'),
                "extracted_medicine": ext.get('medicine', '-'),
                "medicine_match": "✅" if match.medicine_match else "❌",
                "expected_quantity": exp.get('quantity', '-'),
                "extracted_quantity": ext.get('quantity', '-'),
                "quantity_match": "✅" if match.quantity_match else "❌",
                "expected_dosage": exp.get('dosage', '-'),
                "extracted_dosage": ext.get('dosage', '-'),
                "dosage_match": "✅" if match.dosage_match else "❌",
                "match_type": match.match_type
            })
        
        return pd.DataFrame(rows)
