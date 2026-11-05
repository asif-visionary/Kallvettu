# 🏛️ ANCIENT INSCRIPTION RESTORATION & TRANSLATION (கல்வெட்டு மீட்டமைப்பு & மொழிபெயர்ப்பு)

> **An Epigraphical Computer Vision & Classical Tamil NLP Translation Web Application**  
> Decipher, restore, transcribe, and translate ancient stone inscriptions (*Kalvettu* - கல்வெட்டு) across Tamil-Brahmi, Vatteluttu, Grantha, and Medieval Chola scripts.

---

## 🌟 Overview & Key Task

When an ancient **Kalvettu (stone inscription)** image is provided as input, this web application:
1. **Digitally Restores** the weathered stone surface (reconstructing chiseled grooves, removing granite grain, producing archaeological paper squeeze rubbings).
2. **Deciphers & Inpaints** damaged/weathered inscriptional text using classical Tamil morphology and n-gram epigraphical models.
3. **Translates** the ancient inscription into:
   - 📜 **Contemporary Modern Tamil Translation (இக்காலத் தமிழ் மொழிபெயர்ப்பு)**
   - 🌐 **Fluent English Translation (ஆங்கில மொழிபெயர்ப்பு)**
4. **Explains Detailed Meanings**:
   - 📖 **Tamil Meaning & Cultural Background (விரிவான தமிழ் வரலாற்று விளக்கம்)**
   - 🔍 **English Meaning & Epigraphical Commentary (விரிவான தொல்லியல் விளக்கம்)**
5. **Generates Word-by-Word Lexical Analysis (சொல்-பொருள் விளக்கம்)** with root words, grammatical cases, and etymological notes.
6. **Exports** a complete academic Epigraphical Research Dossier (.MD and .JSON).

---

## 📸 Key Features

### 1. 🔬 Multi-Spectral Digital Restoration Studio
- **Luminance Separation (YCbCr):** Enhances stone carving relief without color distortion.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Amplifies sub-millimeter chiseled strokes on low-contrast stones.
- **Sauvola Adaptive Binarization:** Gold standard for ancient epigraphy and document thresholding.
- **Digital Squeeze Rubbing (மை படி):** Simulates the traditional Archaeological Survey of India (ASI) wet paper and charcoal ink squeeze rubbing in three modes:
  - *Traditional Charcoal on Ivory Paper*
  - *Inverted Relief Stone (கருங்கல் புடைப்பு)*
  - *Royal Golden Epigraph (தங்க எழுத்துப் பொறிப்பு)*
- **Chisel Groove Extraction:** Sobel gradient magnitude mapping isolating three-dimensional sculptor chisel marks.
- **Erosion & Weathering Defect Heatmap:** Pinpoints missing or fractured surface sections.

### 2. 🧠 Dual AI Architecture
- **Built-in Offline Epigraphy Engine:** Deterministic morphological analyzer, classical Tamil Sandhi splitter, epigraphical knowledge base with iconic South Indian historical inscriptions.
- **Gemini Vision Multimodal AI (Live API):** Connect your Google Gemini API key to run zero-shot epigraphical decipherment on custom field photos taken directly on smartphones or in museums.

### 3. 🏛️ Curated Sample Inscriptions Library (1-Click Testing)
The app comes pre-packaged with 6 authentic stone inscription simulations:
1. **மாங்குளம் குகைக் கல்வெட்டு (Mangulam Inscription):** Early Pandya King Neduncheliyan (3rd c. BCE, Tamil-Brahmi).
2. **தஞ்சைப் பெரிய கோயில் கல்வெட்டு (Thanjavur Brihadisvara Inscription):** Emperor Rajaraja Chola I (1010 CE, Medieval Chola Tamil).
3. **புகழூர் சேரர் கல்வெட்டு (Pugalur Inscription):** Chera King Athan Cheral Irumporai genealogy (1st-2nd c. CE, Tamil-Brahmi).
4. **பூலாங்குறிச்சி கல்வெட்டு (Pulankurichi Inscription):** King Chendan Kurran land rights (5th c. CE, Early Vatteluttu).
5. **செங்கம் நடுகல் கல்வெட்டு (Chengam Hero Stone):** Memorial for warrior fallen defending village cattle (8th c. CE, Vatteluttu).
6. **உத்திரமேரூர் குடவோலை கல்வெட்டு (Uttiramerur Inscription):** Democratic pot-ballot election constitution (920 CE, Parantaka Chola I).

---

## 📂 Project Structure

```
NLP FINAL/
│
├── app.py                      # Streamlit Web Application (Full UI & Dashboard)
├── restoration_engine.py       # Computer Vision Pipeline (CLAHE, Sauvola, Squeeze, Grooves)
├── nlp_translation_engine.py   # Archaic Tamil NLP, Restoration, Lexicon & Translators
├── multimodal_ai.py            # Gemini Vision API Epigraphy Integration
├── sample_generator.py         # Generates authentic high-res Kalvettu stone images
│
├── assets/
│   ├── styles.css              # Custom Museum & Epigraphy Dark Stone Theme
│   └── samples/                # Generated 6 iconic historical stone inscriptions
│       ├── mangulam.jpg
│       ├── thanjavur.jpg
│       ├── pugalur.jpg
│       ├── pulankurichi.jpg
│       ├── nadukal.jpg
│       └── uttiramerur.jpg
│
├── requirements.txt            # Python dependencies
└── README.md                   # Complete Documentation
```

---

## 🚀 How to Run

### Step 1: Ensure Python is installed
Python 3.10+ is recommended. Check version:
```powershell
python --version
```

### Step 2: Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### Step 3: Launch the Web App
```powershell
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📜 Historical Scripts Supported

| Script Name | Approximate Period | Geological Medium | Distinctive Characteristics |
|---|---|---|---|
| **Tamil-Brahmi (தமிழி)** | 3rd c. BCE – 2nd c. CE | Granite caverns, rock beds | Angular straight strokes, unpointed consonants (no pulli). |
| **Vatteluttu (வட்டெழுத்து)** | 4th c. CE – 11th c. CE | Hero stones (நடுகல்), rocks | Cursive, rounded loops developed for palm-leaf stylus writing. |
| **Grantha (கிரந்தம்)** | 6th c. CE – 9th c. CE | Temple basement walls | Devised for Sanskrit ligatures alongside Tamil phrases. |
| **Medieval Chola Tamil** | 9th c. CE – 13th c. CE | Temple Adhisthana (granite) | Standardized rectangular shapes with horizontal headstrokes. |

---

## 👨‍💻 Developed For
Epigraphy, Computational Linguistics, Archaeology, and Classical Tamil Heritage Preservation.
"# Kallvettu" 
"# Kallvettu" 
