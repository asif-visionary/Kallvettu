"""
Multimodal Vision-Language AI Integration for Ancient Inscription Epigraphy.
Enables zero-shot deciphering, damaged stroke reconstruction, and multi-tier translation
using Gemini Vision API, with structured epigraphical schema validation and graceful fallback.
"""

import json
import base64
import io
import re
import requests
from typing import Dict, Any, Optional, List
from PIL import Image

EPIGRAPHY_SYSTEM_PROMPT = """You are an eminent Senior Epigraphist, Dravidian Palaeographer, and Classical Tamil Archaeologist.
Your task is to analyze, restore, transcribe, and translate an ancient stone inscription (கல்வெட்டு - Kalvettu).
The inscription may be written in Tamil-Brahmi (தமிழி/Damili), Vatteluttu (வட்டெழுத்து), Pallava Grantha (கிரந்தம்), or Medieval Chola/Pandya Tamil.

Given the image of the inscription (and its digitally restored high-contrast squeeze rubbing):
1. Identify the epigraphical script, century, dynasty, ruler, and geographical provenance.
2. Provide a Diplomatic Transcription: preserve visible characters, and enclose weathered, broken, or conjecturally reconstructed glyphs in square brackets like [சாத்தன்].
3. Provide a Normalized Inscription: standardize into modern Tamil Unicode characters with proper pulli and ligatures.
4. Provide a Word-Segmented Reading: split compound words (Sandhi splitting / புணர்ச்சி பிரித்தல்).
5. Provide a Contemporary Modern Tamil translation (இக்காலத் தமிழ் உரைநடை மொழிபெயர்ப்பு).
6. Provide an accurate Scholarly English translation.
7. Provide an in-depth Historical & Cultural Commentary in Tamil and English (dynasty, ruler, purpose like land grant, hero stone, temple endowment).
8. Provide a word-by-word lexical and morphological breakdown (root, grammatical POS, case suffix, modern Tamil meaning, English meaning).
9. Identify any Uncertain or Ambiguous Characters with possible alternative readings and confidence scores.

You MUST respond strictly with a valid JSON object matching this exact schema:
{
  "script_identified": {
    "script_ta": "தமிழ் பிராமி / வட்டெழுத்து / கிரந்தம் / இடைக்காலச் சோழர் தமிழ்",
    "script_en": "Tamil-Brahmi / Vatteluttu / Grantha / Medieval Chola Tamil"
  },
  "confidence_score": 0.85,
  "diplomatic_transcription": "Transcription with damaged glyphs in brackets [ ] and unreadable as (.)",
  "normalized_transcription": "Standard Tamil Unicode representation",
  "word_segmented_text": "Sandhi-split space-separated words",
  "modern_tamil_translation": "முழுமையான எளிய நவீன தமிழ் மொழிபெயர்ப்பு",
  "english_translation": "Full scholarly English translation",
  "tamil_meaning": "விரிவான வரலாற்று, சமுதாய மற்றும் சமயப் பின்னணி விளக்கம் தமிழில்",
  "english_meaning": "Comprehensive archaeological, historical, and palaeographical commentary in English",
  "epigraphical_commentary": {
    "period": "Century / Historical Era",
    "dynasty": "Dynasty Name",
    "ruler": "Monarch / Chieftain",
    "purpose": "Hero Stone / Land Grant / Temple Inscription / Jain Bed / Civic Constitution",
    "historical_context": "Detailed background"
  },
  "uncertain_characters": [
    {
      "character": "[damaged_glyph]",
      "position": 1,
      "alternative_readings": ["reading_A", "reading_B"],
      "reason": "Why the character is weathered or ambiguous"
    }
  ],
  "words": [
    {
      "word": "பண்டைய சொல்",
      "root": "பகுதி / மூலச்சொல்",
      "pos": "இலக்கண வகை (Noun, Verb, etc.)",
      "case_marker": "வேற்றுமை உருபு (Nominative, Dative, etc.)",
      "modern": "நவீன தமிழ்ப் பொருள்",
      "english": "English meaning",
      "notes": "இலக்கண அல்லது வரலாற்று குறிப்பு"
    }
  ]
}
"""

SUPPORTED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
]


def image_to_base64(pil_image: Image.Image, format="JPEG") -> str:
    """Converts a PIL image to a base64 encoded string, optimizing dimensions for network transfer."""
    buffered = io.BytesIO()
    rgb_img = pil_image.convert("RGB")
    rgb_img.thumbnail((1400, 1400))
    rgb_img.save(buffered, format="JPEG", quality=88)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Cleans markdown code fences and extracts a valid JSON dictionary."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse valid JSON from model response. Raw: {text[:200]}...")


def validate_gemini_epigraphy_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes Gemini Vision response against the epigraphical schema.
    Ensures backward compatibility with workbench UI fields while standardizing multi-tier attributes.
    """
    if not isinstance(data, dict):
        raise ValueError("Model response must be a JSON object dictionary.")

    # Script details
    script_info = data.get("script_identified", {})
    if isinstance(script_info, dict):
        script_ta = script_info.get("script_ta", data.get("script_ta", "பண்டைய தமிழ் எழுத்து"))
        script_en = script_info.get("script_en", data.get("script_en", "Ancient Tamil Epigraphical Script"))
    else:
        script_ta = data.get("script_ta", str(script_info))
        script_en = data.get("script_en", str(script_info))

    # Transcriptions
    diplomatic = data.get("diplomatic_transcription", data.get("weathered_text", ""))
    normalized = data.get("normalized_transcription", data.get("restored_text", diplomatic))
    segmented = data.get("word_segmented_text", normalized)

    # Commentary
    commentary = data.get("epigraphical_commentary", {})
    if not isinstance(commentary, dict):
        commentary = {}

    era_ta = commentary.get("period", data.get("era_ta", "வரலாற்றுக் காலம்"))
    era_en = commentary.get("period", data.get("era_en", "Historical Era"))
    dynasty_ta = commentary.get("dynasty", data.get("dynasty_ta", "பண்டைய தமிழகம்"))
    dynasty_en = commentary.get("dynasty", data.get("dynasty_en", "Ancient Tamil Horizon"))
    category = commentary.get("purpose", data.get("category", "கல்வெட்டு ஆவணம் (Historical Epigraph)"))

    # Words
    raw_words = data.get("words", [])
    normalized_words = []
    if isinstance(raw_words, list):
        for w in raw_words:
            if isinstance(w, dict):
                normalized_words.append({
                    "word": w.get("word", ""),
                    "root": w.get("root", w.get("word", "")),
                    "pos": w.get("pos", w.get("grammar", "பெயர்ச்சொல் (Noun)")),
                    "case_marker": w.get("case_marker", "வேற்றுமை உருபற்ற எழுவாய் (Nominative)"),
                    "modern": w.get("modern", w.get("modern_meaning", "")),
                    "english": w.get("english", w.get("english_meaning", "")),
                    "notes": w.get("notes", "")
                })

    # Uncertain characters
    uncertain_chars = data.get("uncertain_characters", [])
    if not isinstance(uncertain_chars, list):
        uncertain_chars = []

    return {
        "id": "gemini_decipherment",
        "title_ta": f"தொல்லியல் ஆய்வு: {dynasty_ta}",
        "title_en": f"Epigraphical Analysis: {dynasty_en}",
        "script_ta": script_ta,
        "script_en": script_en,
        "script_identified": {"script_ta": script_ta, "script_en": script_en},
        "era_ta": era_ta,
        "era_en": era_en,
        "dynasty_ta": dynasty_ta,
        "dynasty_en": dynasty_en,
        "category": category,
        "confidence_score": float(data.get("confidence_score", 0.85)),
        "diplomatic_text": diplomatic,
        "normalized_text": normalized,
        "word_segmented_text": segmented,
        "weathered_text": diplomatic,
        "restored_text": normalized,
        "modern_tamil_translation": data.get("modern_tamil_translation", ""),
        "english_translation": data.get("english_translation", ""),
        "tamil_meaning": data.get("tamil_meaning", ""),
        "english_meaning": data.get("english_meaning", ""),
        "epigraphical_commentary": commentary,
        "uncertain_characters": uncertain_chars,
        "words": normalized_words,
        "provenance": "Gemini Multimodal Epigraphical AI",
        "data_source": "Gemini Multimodal Vision API"
    }


def call_gemini_vision_api(
    image: Image.Image,
    api_key: str,
    restored_image: Optional[Image.Image] = None,
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Calls the Gemini Vision REST API for zero-shot epigraphical decipherment.
    Sends original stone image and supplementary high-contrast squeeze rubbing.
    Enforces structured JSON schema validation and provides diagnostic error handling.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Gemini API key is required. Please set it in Settings Studio or via GEMINI_API_KEY environment variable.")

    clean_model = model_name.replace("models/", "").strip()
    if clean_model in ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-flash"):
        clean_model = "gemini-2.5-flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key.strip()}"

    headers = {"Content-Type": "application/json"}

    user_parts: List[Dict[str, Any]] = [
        {"text": "Please decipher, restore, transcribe, and translate this stone inscription according to your archaeological and epigraphical instructions."}
    ]

    # Add original stone photograph
    b64_orig = image_to_base64(image)
    user_parts.append({
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": b64_orig
        }
    })

    # If digital squeeze rubbing is available, include it as supplementary archaeological squeeze
    if restored_image:
        b64_rest = image_to_base64(restored_image)
        user_parts.append({
            "text": "The following is the high-contrast digital squeeze rubbing (எபிகிராபி மை படி) of the above stone inscription:"
        })
        user_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64_rest
            }
        })

    payload = {
        "systemInstruction": {
            "parts": [{"text": EPIGRAPHY_SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": user_parts
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        raise RuntimeError("Gemini API request timed out after 60 seconds. Check network connectivity or retry.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error connecting to Gemini API: {e}")

    # Handle HTTP status codes
    if response.status_code == 429:
        raise RuntimeError("Gemini API quota exceeded (HTTP 429). Please verify your API tier or retry after a cooldown.")
    elif response.status_code in [400, 403]:
        raise RuntimeError(f"Gemini API authentication failed (HTTP {response.status_code}): {response.text}")
    elif response.status_code != 200:
        raise RuntimeError(f"Gemini API returned unexpected status {response.status_code}: {response.text}")

    resp_json = response.json()
    candidates = resp_json.get("candidates", [])
    if not candidates:
        feedback = resp_json.get("promptFeedback", {})
        raise RuntimeError(f"No response candidates returned. Prompt feedback: {feedback}")

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "UNKNOWN")
    if finish_reason not in ["STOP", "MAX_TOKENS"]:
        raise RuntimeError(f"Gemini generation stopped unexpectedly with reason: {finish_reason}")

    try:
        candidate_text = candidate["content"]["parts"][0]["text"]
        raw_dict = clean_json_response(candidate_text)
        validated = validate_gemini_epigraphy_response(raw_dict)
        return validated
    except Exception as e:
        raise RuntimeError(f"Failed to parse and validate Gemini response as structured epigraphy JSON: {e}\nRaw output: {candidate_text if 'candidate_text' in locals() else resp_json}")
