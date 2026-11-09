# 📜 Historical Script Recognition Research Architecture (வரலாற்று எழுத்தாய்வு கட்டமைப்பு)

## 1. Research Objectives & Motivation
Modern commercial and open-source OCR engines (such as Tesseract, Google Cloud Vision, and PaddleOCR PP-OCRv5) are trained exclusively on modern printed and digital Unicode Tamil text. They lack the morphological, typographic, and palaeographical weights necessary to recognize ancient epigraphical scripts chiseled onto cavern overhangs, hero stones, and granite temple basements:
- **Tamil-Brahmi (தமிழி / தாமிழி):** c. 3rd Century BCE – 2nd Century CE (e.g. Mangulam, Pugalur, Jambai).
- **Vatteluttu (வட்டெழுத்து):** c. 4th Century CE – 11th Century CE (e.g. Pulankurichi, Chengam, Thirunandikkara).
- **Pallava & Chola Grantha (கிரந்தம்):** Used for Sanskrit formulas, royal prasastis, and ligatures in bilingual epigraphs.
- **Medieval Chola Tamil:** c. 9th – 13th Century CE (e.g. Thanjavur, Uttiramerur).

Kallvettu establishes a scientifically defensible, verifiable research architecture for historical epigraphical script recognition without making fraudulent claims of "100% automated ancient decipherment."

---

## 2. Academic Dataset Ingestion & Metadata Standards

### 2.1 The Mendeley Tamil Brahmi Stone Inscription Dataset
- **Citation:** *P. Subashree et al., "Tamil Brahmi Stone Inscription Dataset," Mendeley Data, V1, 2021.*
- **DOI:** [10.17632/2spz2js9jp.1](https://data.mendeley.com/datasets/2spz2js9jp/1)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Contents:** Chiseled Tamil-Brahmi character crops across 32 signary classes extracted from natural rock shelters in Madurai, Karur, and Pudukkottai districts.

### 2.2 In-Repository Dataset Directory Organization
The research workbench organizes historical corpora under [`datasets/`](file:///c:/Users/Mohamed%20Asif/OneDrive/Desktop/Basic_Projects/Kallvettu/datasets):
```text
datasets/
├── tamil_brahmi/
│   ├── original_images/       # High-resolution archaeological field photographs
│   ├── line_crops/             # Rectified and deskewed text lines
│   ├── character_crops/        # Normalized glyph crops
│   ├── annotations/            # Spatial XML / JSON bounding polygons
│   └── ground_truth/           # Diplomatic transcription references
├── vatteluttu/
│   ├── original_images/
│   ├── line_crops/
│   ├── character_crops/
│   ├── annotations/
│   └── ground_truth/
├── medieval_tamil/
│   ├── original_images/
│   ├── line_crops/
│   ├── character_crops/
│   ├── annotations/
│   └── ground_truth/
└── metadata/
    ├── inscriptions.csv        # Master epigraph metadata, dynasties, locations & licenses
    ├── characters.csv          # Normalized bounding box coordinates (ymin, xmin, ymax, xmax)
    └── annotation_history.csv  # Epigraphist review logs, disputes & decisions
```

---

## 3. Anti-Leakage Data Partitioning Strategy

### The Cave / Monument Isolation Rule
In standard computer vision benchmarks, random shuffling across image patches causes severe data leakage because adjacent character crops from the same rock wall share identical chisel depth, grain texture, lighting, and erosion patterns.

Kallvettu enforces **Site-Level Grouped Splitting** via [`EpigraphicalDatasetManager.generate_site_level_split()`](file:///c:/Users/Mohamed%20Asif/OneDrive/Desktop/Basic_Projects/Kallvettu/services/research/dataset_framework.py):
1. All characters originating from a single inscription/monument (e.g., `TB_MGL_001`) are assigned **exclusively** to Train, Validation, or Test.
2. The Test set tests generalization to *previously unseen caverns and stone surfaces*.
3. Any cross-split overlap in inscription identifiers raises an immediate `ValueError` integrity violation.

---

## 4. Deep Learning Recognition Architectures for Lithic Scripts

### 4.1 SignaryNet (Glyph-Level Dense Residual Classifier)
- **Input:** $64 \times 64 \times 1$ grayscale crops normalized with Sauvola adaptive binarization ($k=0.20, w=25$).
- **Backbone:** Modified ResNet-18 with Coordinate Attention layers to focus on shallow chisel grooves while ignoring granite quartz flecks.
- **Output:** Categorical probability over the 32 Tamil-Brahmi glyph inventory + 1 `ambiguous/damaged` class.

### 4.2 DR-LIFT (Dense Residual Local Invariant Feature Transformer)
- Sequence-level recognition combining 1D positional encodings with CTC (Connectionist Temporal Classification) loss for unsegmented text lines.
- Eliminates the error propagation inherent in manual character segmentation.

---

## 5. Epigraphical Provenance & Scientific Disclaimers
1. **No Simulated Ground Truth:** Machine learning models must never be evaluated against AI-generated transcriptions. Ground truth in `inscriptions.csv` is derived solely from published Archaeological Survey of India (ASI) *Epigraphia Indica* volumes and state archaeological survey monographs.
2. **Uncertainty Preservation:** Glyphs with damaged strokes are encoded with multiple valid alternative readings (e.g. `saa | si | ci` for weathered Brahmic ligatures) rather than forced into a single modern character.
3. **Researcher Control:** Model inferences are surfaced as *hypotheses* for the epigraphist's review and approval.

