# ASR Prompt Engineering Guide for Pharma Voice Orders

> **Purpose**: Define expected voice order formats, sample patterns, and entity schema to improve transcription accuracy and structured data extraction.

---

## Expected Order Format

The ASR model should recognize **medicine orders** in the following patterns:

### Pattern 1: Medicine First
```
<Medicine Name> <Form> <Quantity> <Unit>
```
**Example**: "Paracetamol tablet 300 strips"

### Pattern 2: Form First
```
<Form> <Medicine Name> <Quantity> <Unit>
```
**Example**: "Tablet Paracetamol 300 strips"

### Pattern 3: Quantity First
```
<Quantity> <Unit> <Medicine Name> [<Dosage>]
```
**Example**: "20 strips Augmentin 625"

### Pattern 4: Comma-Separated List
```
<Order1>, <Order2>, <Order3>
```
**Example**: "Paracetamol 100 strips, Metformin 50 strips, Crocin 30 strips"

### Pattern 5: Connector Words
```
<Order1> and/also/plus/then <Order2>
```
**Example**: "Send Paracetamol 100 also Metformin 50"

---

## Entity Schema

Each extracted order should contain:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `medicine` | string | Medicine name (as heard) | "Paracetamol" |
| `form` | string | Tablet, Syrup, Injection, etc. | "tablet" |
| `quantity` | string | Number + Unit | "300 strips" |
| `dosage` | string | Strength (mg, ml, etc.) | "500mg" |

---

## Sample Voice Orders (for Testing)

### Simple Orders
1. "Send 20 strips of Augmentin 625"
2. "Paracetamol tablet 100 strips"
3. "Tablet Metformin 500mg 50 strips"
4. "Order Crocin 650 30 strips"
5. "50 bottles of Ascoril syrup"
6. "20 tubes of Betnovate cream"
7. "10 vials of Amikacin injection"

### Multi-Item Orders
1. "Paracetamol 100 strips, Metformin 50 strips, Crocin 30 strips"
2. "Send Augmentin 20 strips also Calpol 15 strips and Dolo 10 strips"
3. "I need 50 Azithromycin, 30 Cetirizine, and 20 Omez"

### Complex/Noisy Orders
1. "Uh, send me Paracetamol, maybe 100? And also some Metformin"
2. "Tablet Paraacetamole 300 slips" (misspelling of Paracetamol, slips instead of strips)
3. "Give me twenty strips of Aug-mentin six two five"

---

## Form Keywords

The model should recognize these form indicators:

| Form Type | Keywords |
|-----------|----------|
| Tablet | tablet, tab, tabs, capsule, cap, caps |
| Syrup | syrup, liquid, bottle, suspension |
| Injection | injection, inj, vial, ampoule |
| Cream/Gel | cream, gel, ointment, tube |
| Spray | spray, inhaler, puff |
| Drops | drops, eye drops, ear drops |
| Sachet | sachet, powder, granules |

---

## Unit Keywords

| Unit Type | Keywords |
|-----------|----------|
| Strips | strips, strip, slips, slip |
| Bottles | bottles, bottle, btl |
| Tablets | tablets, tabs, pieces, pcs |
| Boxes | boxes, box, packs, pack |
| Vials | vials, vial, ampoules |

---

## Common Pronunciation Variations

| Correct Name | Common Variations |
|--------------|-------------------|
| Paracetamol | paraacetamole, parcetamol, paracetmal |
| Metformin | metformine, metforman, metphormin |
| Augmentin | augmentine, agmentin, augmuntin |
| Azithromycin | azithromicin, azithro, azith |
| Cetirizine | cetirizin, cetrizine, cetriz |
| Pantoprazole | pantop, pantoprazol |

---

## Structured Output Target

After processing, each order should be structured as:

```json
{
  "medicine": "Paracetamol",
  "medicine_standardized": "Crocin",  // Matched from DB
  "form": "tablet",
  "quantity": "300 strips",
  "dosage": "650mg",
  "manufacturer": "Sun Pharma",
  "original_segment": "Paracetamol tablet 300 strips"
}
```

---

## Tips for Model Training

1. **Normalize Numbers**: Convert "twenty" → 20, "hundred" → 100
2. **Handle Filler Words**: Ignore "uh", "um", "like", "maybe"
3. **Fuzzy Match Medicine Names**: Use 80%+ confidence threshold
4. **Default Values**: If unit not specified, use DB default
5. **Case Insensitive**: All matching should be lowercase

