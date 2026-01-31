"""
Ground Truth Generator for Evaluation

Generates ground truth data from Medicine_Names_Packges.csv for evaluation purposes.
This creates a representative sample of medicines with expected extraction values.
"""

import pandas as pd
import random
from pathlib import Path


def extract_dosage_from_name(medicine_name: str) -> str:
    """Extract dosage from medicine name like 'ACTAVIR-400MG' -> '400mg'."""
    import re
    # Look for patterns like -400MG, 400MG, 400 MG, etc.
    match = re.search(r'(\d+)\s*(MG|MCG|ML|GM)', medicine_name.upper())
    if match:
        return f"{match.group(1)}{match.group(2).lower()}"
    return "-"


def extract_form_from_packing(packing: str) -> str:
    """Extract form from packing like '1*10TAB' -> 'tablet'."""
    packing_upper = str(packing).upper()
    if 'TAB' in packing_upper or 'CAP' in packing_upper:
        return 'tablet'
    elif 'SYP' in packing_upper or 'ML' in packing_upper:
        return 'syrup'
    elif 'DROPS' in packing_upper:
        return 'drops'
    elif 'INJ' in packing_upper:
        return 'injection'
    elif 'CREAM' in packing_upper or 'GEL' in packing_upper:
        return 'cream'
    elif 'SPRAY' in packing_upper:
        return 'spray'
    return 'tablet'  # Default


def generate_quantity() -> str:
    """Generate a random realistic quantity."""
    quantities = [
        '10 strips', '20 strips', '50 strips', '100 strips',
        '5 bottles', '10 bottles', '20 bottles',
        '10 packs', '20 packs', '50 packs',
        '5 boxes', '10 boxes',
        '10 vials', '20 vials'
    ]
    return random.choice(quantities)


def generate_ground_truth(
    input_csv: str = "data/Medicine_Names_Packges.csv",
    output_csv: str = "evaluation/ground_truth.csv",
    medicines_per_audio: int = 3
):
    """
    Generate ground truth CSV from medicine database.
    Assigns all medicines across available audio files.
    
    Args:
        input_csv: Path to Medicine_Names_Packges.csv
        output_csv: Path to output ground_truth.csv
        medicines_per_audio: Number of medicines to assign per audio file
    """
    from pathlib import Path
    
    # Load medicine data
    df = pd.read_csv(input_csv)
    
    # Clean and prepare data
    df = df.rename(columns={
        "ITEM NAME": "medicine_name",
        "Full_Name_of_Manufacturer": "manufacturer",
        "PACKING": "packing"
    })
    
    # Remove rows with missing medicine names
    df = df.dropna(subset=['medicine_name'])
    
    # Get all audio files from audioData directory
    audio_dir = Path("audioData")
    if audio_dir.exists():
        audio_files = sorted([f.name for f in audio_dir.iterdir()
                              if f.suffix in ['.m4a', '.wav', '.mp3', '.ogg']])
    else:
        audio_files = [f"R_{i:03d}.m4a" for i in range(1, 21)]  # Default to R_001 to R_020
    
    if not audio_files:
        audio_files = [f"R_{i:03d}.m4a" for i in range(1, 21)]
    
    ground_truth_rows = []
    order_index = 1
    
    # Distribute all medicines across audio files
    all_medicines = df.to_dict('records')
    num_audio_files = len(audio_files)
    
    for idx, row in enumerate(all_medicines):
        # Assign medicine to an audio file (round-robin distribution)
        audio_file = audio_files[idx % num_audio_files]
        
        medicine_name = str(row['medicine_name']).strip()
        packing = str(row.get('packing', ''))
        mfr = str(row.get('manufacturer', 'Unknown')).strip()
        
        # Extract dosage from name
        dosage = extract_dosage_from_name(medicine_name)
        
        # Extract form from packing
        form = extract_form_from_packing(packing)
        
        # Generate realistic quantity
        quantity = generate_quantity()
        
        ground_truth_rows.append({
            'audio_file': audio_file,
            'order_index': order_index,
            'medicine_name': medicine_name,
            'quantity': quantity,
            'dosage': dosage,
            'form': form,
            'manufacturer': mfr
        })
        order_index += 1
    
    # Create ground truth DataFrame
    gt_df = pd.DataFrame(ground_truth_rows)
    
    # Save to CSV with header comment
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write with comment header
    with open(output_path, 'w') as f:
        f.write("# Ground Truth Dataset for Entity Extraction Evaluation\n")
        f.write("# Generated from Medicine_Names_Packges.csv\n")
        f.write("# Instructions: This is auto-generated ground truth for evaluation\n")
        f.write("# Columns: audio_file, order_index, medicine_name, quantity, dosage, form, manufacturer\n")
        gt_df.to_csv(f, index=False)
    
    print(f"Generated ground truth with {len(gt_df)} entries")
    print(f"Covering {df['manufacturer'].nunique()} manufacturers")
    print(f"Distributed across {num_audio_files} audio files")
    print(f"Saved to: {output_path}")
    
    # Print sample
    print("\nSample entries:")
    print(gt_df.head(10).to_string())
    
    return gt_df


def generate_test_scenarios(
    input_csv: str = "data/Medicine_Names_Packges.csv",
    output_json: str = "evaluation/test_cases.json",
    num_scenarios: int = 50
):
    """
    Generate test scenarios with expected medicines for testing.
    
    Args:
        input_csv: Path to Medicine_Names_Packges.csv
        output_json: Path to output test_cases.json
        num_scenarios: Number of test scenarios to generate
    """
    import json
    
    # Load medicine data
    df = pd.read_csv(input_csv)
    df = df.rename(columns={
        "ITEM NAME": "medicine_name",
        "Full_Name_of_Manufacturer": "manufacturer",
        "PACKING": "packing"
    })
    df = df.dropna(subset=['medicine_name'])
    
    test_cases = []
    
    # Generate single medicine orders
    for i in range(min(num_scenarios // 2, len(df))):
        row = df.iloc[i]
        medicine_name = str(row['medicine_name']).strip()
        quantity = generate_quantity()
        dosage = extract_dosage_from_name(medicine_name)
        
        # Create natural language text
        templates = [
            f"Send me {quantity} of {medicine_name}",
            f"I need {quantity} of {medicine_name}",
            f"Order {quantity} {medicine_name}",
            f"Give me {quantity} of {medicine_name}",
            f"Add {quantity} {medicine_name} to the order"
        ]
        
        test_cases.append({
            "text": random.choice(templates),
            "expected": [{
                "medicine": medicine_name,
                "quantity": quantity,
                "dosage": dosage
            }]
        })
    
    # Generate multi-medicine orders
    for i in range(num_scenarios // 2):
        # Pick 2-3 random medicines
        num_meds = random.randint(2, 3)
        sampled = df.sample(n=num_meds, random_state=42+i)
        
        medicines = []
        text_parts = []
        
        for _, row in sampled.iterrows():
            medicine_name = str(row['medicine_name']).strip()
            quantity = generate_quantity()
            dosage = extract_dosage_from_name(medicine_name)
            
            medicines.append({
                "medicine": medicine_name,
                "quantity": quantity,
                "dosage": dosage
            })
            text_parts.append(f"{quantity} of {medicine_name}")
        
        # Create natural language text
        templates = [
            f"Send me {', '.join(text_parts[:-1])} and {text_parts[-1]}",
            f"I need {', '.join(text_parts[:-1])} also {text_parts[-1]}",
            f"Order {', '.join(text_parts[:-1])} plus {text_parts[-1]}",
        ]
        
        test_cases.append({
            "text": random.choice(templates),
            "expected": medicines
        })
    
    # Save test cases
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"\nGenerated {len(test_cases)} test scenarios")
    print(f"Saved to: {output_path}")
    
    return test_cases


if __name__ == "__main__":
    # Generate ground truth CSV with ALL medicines
    gt_df = generate_ground_truth(
        input_csv="data/Medicine_Names_Packges.csv",
        output_csv="evaluation/ground_truth.csv"
    )
    
    # Generate test scenarios JSON
    test_cases = generate_test_scenarios(
        input_csv="data/Medicine_Names_Packges.csv",
        output_json="evaluation/test_cases.json",
        num_scenarios=100
    )
