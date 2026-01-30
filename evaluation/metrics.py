import jiwer
from rapidfuzz import fuzz


class MetricsEvaluator:
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate."""
        if not reference or not hypothesis:
            return 1.0
        return jiwer.wer(reference, hypothesis)

    @staticmethod
    def calculate_entity_accuracy(expected_entities: list, extracted_entities: list) -> float:
        """
        Calculate accuracy of extracted entities vs ground truth.
        Simple logic: (matches / total_expected)
        """
        if not expected_entities:
            return 0.0
            
        matches = 0
        for exp in expected_entities:
            # Check if this expected medicine was found in extracted list
            found = False
            for ext in extracted_entities:
                if fuzz.ratio(exp['medicine'].lower(), ext['medicine'].lower()) > 85:
                    found = True
                    break
            if found:
                matches += 1
                
        return matches / len(expected_entities)
