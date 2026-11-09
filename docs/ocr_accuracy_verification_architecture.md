# Kallvettu — OCR Accuracy & Multi-Engine Palaeographical Verification Architecture

**Version:** 3.0.0-verified  
**Domain:** Computational Epigraphy & Digital Heritage Preservation  
**Tamil Name:** கல்வெட்டு எழுத்துப் படிம பகுப்பாய்வு மற்றும் பல-இயந்திர சரிபார்ப்பு நெறிமுறை  

---

## 1. Overview & Problem Definition

In South Indian stone epigraphy, automated optical character recognition faces severe unique challenges:
1. **Physical surface degradation**: Granite crystal flecks, lichens, erosion pits, and incomplete chisel strikes.
2. **Reading order ambiguities**: Irregular line spacing, damaged margins, and boustrophedon or multi-register layout.
3. **Palaeographical script evolution**: Tamil-Brahmi (3rd c. BCE – 3rd c. CE), Early Vatteluttu (4th – 8th c. CE), Classical Chola Tamil (9th – 13th c. CE), and Grantha ligatures.
4. **False dispute cascading**: Discrepant tokenization across competing OCR engines (e.g. `['ஸ்வஸ்தி', 'ஸ்ரீ']` vs `['ஸ்வஸ்திஸ்ரீ']`) caused traditional positional index-matching to flag 100% of subsequent valid words as disputed, yielding artificial 5.6% agreement rates.

The Kallvettu verification engine eliminates these limitations through cryptographic source identity, adaptive preprocessing variants, reading-order line clustering, and sequence alignment.

---

## 2. Source Data Isolation & Provenance

Every analyzed stone inscription receives a deterministic `SourceIdentity`:

```
Source Identity Schema:
├── source_image_id: "SRC_" + SHA-256[:12]
├── image_sha256: Hex digest of pristine pixel buffer
├── original_filename: Upload filename or catalog preset reference
├── session_id: UUIDv4 of active research session
├── ocr_execution_id: "OCR_" + UUIDv4[:8]
├── provenance_mode:
│   ├── curated_preset: Validated catalog ground truth (e.g., Mahadevan 2003)
│   ├── live_ocr: Raw computational inference from computer vision/AI models
│   └── manual_researcher: Scholar-edited diplomatic revision
└── created_at: High-resolution timestamp
```

### Invariant:
OCR readings are **never compared across disparate image hashes**. Curated catalog ground-truth presets are **never silently substituted** for live computational OCR predictions unless the researcher explicitly activates preset matching.

---

## 3. Seven Independent Preprocessing Variants

Simulated high-pass charcoal squeeze rubbings (`"squeeze"`) introduce substantial high-frequency artificial texture and stroke dilation that degrade deep learning OCR. The pipeline now isolates 7 distinct image forms:

| Variant ID | Name | Method & Parameters | Default for OCR? |
| :--- | :--- | :--- | :--- |
| `original` | Original Pristine | Unaltered RGB pixels | No |
| `grayscale` | BT.601 Grayscale | Normalized luminance channel | No |
| **`mild_clahe`** | **Mild CLAHE** | **Luminance YCbCr equalization (clip 1.8, blend 0.85)** | **YES (Primary)** |
| `denoised_gray` | Denoised Grayscale | Bilateral & median texture smoothing | Optional fallback |
| `sauvola` | Sauvola Binarization | Dynamic local variance thresholding ($k=0.20, W=25$) | Visual inspection |
| `adaptive_thresh` | Adaptive Gaussian | Local Gaussian neighborhood thresholding ($C=0.03$) | Visual inspection |
| `roi_crop` | Region of Interest | Bounding box focused incision zone | Targeted word check |

---

## 4. PaddleOCR Reading Order & Line Clustering

The local offline OCR engine utilizes official PaddleOCR 3.x with:
- **Detection Model:** `PP-OCRv6_medium_det`
- **Tamil Recognition Model:** `ta_PP-OCRv5_mobile_rec`

### Two-Level Reading Order Sorting Algorithm:
1. Parse detected bounding polygons $P_i$ and compute vertical bounding box centers $y_{c, i}$ and heights $h_i$.
2. Sort bounding boxes initially by $y_c$.
3. Cluster into lines: Box $B_k$ joins line $L_m$ if $|y_{c, k} - \bar{y}_{c, L_m}| \le 0.5 \times \bar{h}_{L_m}$.
4. Sort line clusters top-to-bottom: $\bar{y}_{c, L_1} < \bar{y}_{c, L_2} < \dots$
5. Sort tokens within each line left-to-right: $x_{\min, 1} < x_{\min, 2} < \dots$

This prevents word transposition between adjacent registers and preserves natural epigraphical reading sequence.

---

## 5. Sequence Alignment & Inter-Engine Agreement

### Unicode NFC Normalization
All candidate transcriptions undergo canonical Unicode NFC composition and invisible character stripping:
$$\text{Clean}(T) = \text{RegexStrip}\Big(\text{unicodedata.normalize}(\text{"NFC"}, T)\Big)$$

### Sequence Matching (No Cascading Index Shifts)
Instead of comparing array indices $0, 1, 2, \dots$:
$$\text{Matcher} = \text{SequenceMatcher}(\text{None}, \text{Tokens}_A, \text{Tokens}_B)$$

Opcodes are classified into palaeographical dispute categories:
- `'equal'`: Both engines agree on glyph sequence $\rightarrow$ **Agreed Token**
- `'replace'` (same length): Model A proposes $W_A$ vs Model B proposes $W_B$ $\rightarrow$ **Substitution Dispute**
- `'replace'` (different lengths): Token boundary difference (e.g. `'ஸ்வஸ்தி ஸ்ரீ'` vs `'ஸ்வஸ்திஸ்ரீ'`) $\rightarrow$ **Word Boundary Dispute**
- `'delete'`: Token detected by Model A but omitted by Model B $\rightarrow$ **Omission Dispute**
- `'insert'`: Token omitted by Model A but detected by Model B $\rightarrow$ **Insertion Dispute**

### Truthful Metric Labeling:
The inter-engine metric is **never advertised as accuracy**. It is explicitly labeled:
$$\text{Inter-Engine Agreement Rate} = \frac{N_{\text{agreed}}}{N_{\text{agreed}} + N_{\text{disputed}}}$$

---

## 6. Transcription Revision History & Audit Trail

To safeguard raw computational outputs while enabling scholarly correction:
1. Raw OCR results in `results_by_provider` are **immutable** and never overwritten.
2. Researcher edits create a structured audit entry:
   - `revision_id`: Sequential integer
   - `timestamp`: ISO-formatted datetime
   - `editor`: Authorized epigraphist identifier
   - `previous_text`: Unrevised reading
   - `revised_text`: Scholar's restored text
   - `reason`: Epigraphical rationale (e.g., parallel text restoration)
3. Multi-tier representations are dynamically updated:
   - **Tier 1 (Diplomatic):** Preserves `[ ]` damage brackets and illegible glyphs.
   - **Tier 2 (Normalized):** Canonical classical Tamil orthography.
   - **Tier 3 (Word-Segmented):** Continuous stone script split via classical Sandhi rules.

