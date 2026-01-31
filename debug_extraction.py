
import sys
import os
sys.path.append(os.getcwd())

from simulation.manufacturer_db import ManufacturerDB
from core.entity_extractor import EntityExtractor

print("Loading DB...")
db = ManufacturerDB(data_dir="data")
print(f"DB Medicines: {len(db.medicines)}")
print(f"Sample DB entry: {db.medicines.iloc[0]['medicine_name']}")

extractor = EntityExtractor(db)


from rapidfuzz import process, fuzz

print("Testing Fuzzy Match...")
query = "paracetamol 500 milligram 100 strips"
choices = ["Paracetamol", "Augmentin", "Dolo 650"]

print(f"Query: '{query}'")
match = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
print(f"Match (token_set_ratio): {match}")

match_partial = process.extractOne(query, choices, scorer=fuzz.partial_ratio)
print(f"Match (partial_ratio): {match_partial}")

# Test with actual Extractor again if simple test passes
# ...
