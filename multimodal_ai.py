"""
Multimodal Vision-Language AI Integration for Ancient Inscription Epigraphy.
Enables zero-shot deciphering, damaged stroke reconstruction, and dual-language translation
using Gemini Vision API, with graceful fallback to the local Epigraphy Knowledge Engine.
"""

import json
import base64
import io
import requests
from typing import Dict, Any, Optional
from PIL import Image

EPIGRAPHY_SYSTEM_PROMPT = """You are an eminent Epigraphist, Palaeographer, and Classical Tamil Archaeologist.
Your task is to analyze, restore, transcribe, and translate an ancient stone inscription (கல்வெட்டு - Kalvettu).
The inscription may be written in Tamil-Brahmi (தமிழி/Damili), Vatteluttu (வட்டெழுத்து), Pallava Grantha (கிரந்தம்), or Medieval Chola/Pandya Tamil.

Given the image of the inscription (and its digitally restored rubbing):
1. Identify the epigraphical script, century, dynasty, and geographical provenance.
2. Carefully decipher the chiseled text. If any character is weathered or missing, reconstruct it based on classical Tamil grammatical and epigraphical context, marking restored characters in brackets like [சாத்தன்].
3. Provide the full restored inscription text in Tamil script.
4. Provide a clear, natural Contemporary Modern Tamil translation (இக்காலத் தமிழ் உரைநடை மொழிபெயர்ப்பு).
5. Provide a fluent, accurate English translation.
6. Provide an in-depth Historical & Semantic Explanation in Tamil (விரிவான தமிழ் விளக்கம்).
7. Provide an in-depth Archaeological & Historical Commentary in English (cultural context, dynasty, king, purpose like land-grant, hero stone, temple donation).
8. Provide a word-by-word lexical breakdown (சொல்-பொருள் விளக்கம்).

You MUST respond strictly with a valid JSON object with the following schema:
{
  "script_ta": "தமிழ் பிராமி / வட்டெழுத்து / சோழர் தமிழ்...",
  "script_en": "Tamil-Brahmi / Vatteluttu / Chola Tamil...",
  "era_ta": "கி.மு. / கி.பி. நூற்றாண்டு...",
  "era_en": "Approximate Century / Period...",
  "dynasty_ta": "அரசமரபு மற்றும் அரசன்...",
  "dynasty_en": "Dynasty and Monarch...",
  "category": "Hero Stone / Land Grant / Temple Inscription / Jain Bed...",
  "weathered_text": "Deciphered text with damaged parts in brackets [ ]",
  "restored_text": "Clean restored inscription in Tamil script",
  "modern_tamil_translation": "முழுமையான எளிய நவீன தமிழ் மொழிபெயர்ப்பு",
  "english_translation": "Full grammatical English translation",
  "tamil_meaning": "விரிவான வரலாற்று, சமுதாய மற்றும் சமயப் பின்னணி விளக்கம் தமிழில்",
  "english_meaning": "Comprehensive archaeological, historical, and palaeographical commentary in English",
  "words": [
    {
      "word": "பண்டைய சொல்",
      "modern": "நவீன தமிழ்ப் பொருள்",
      "english": "English meaning",
      "notes": "இலக்கண அல்லது வரலாற்று குறிப்பு"
    }
  ]
}
"""

def image_to_base64(pil_image: Image.Image, format="JPEG") -> str:
    """Converts a PIL image to a base64 encoded string."""
    buffered = io.BytesIO()
    # If image has alpha channel or is 16-bit, convert to RGB
    rgb_img = pil_image.convert("RGB")
    rgb_img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_gemini_vision_api(
    image: Image.Image,
    api_key: str,
    restored_image: Optional[Image.Image] = None,
    model_name: str = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """
    Calls the Gemini Vision API directly using REST API (no external SDK dependency required).
    Sends both original image and restored digital squeeze image for maximum OCR accuracy.
    """
    if not api_key:
        raise ValueError("Gemini API key is required.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}

    # Prepare parts
    parts = [{"text": EPIGRAPHY_SYSTEM_PROMPT}]

    # Add original image
    b64_orig = image_to_base64(image)
    parts.append({
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": b64_orig
        }
    })

    # If digital squeeze rubbing is available, include it as supplementary epigraphic squeeze
    if restored_image:
        b64_rest = image_to_base64(restored_image)
        parts.append({
            "text": "The following is the high-contrast digital squeeze rubbing (எபிகிராபி மை படி) of the above inscription:"
        })
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64_rest
            }
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=45)

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API returned status code {response.status_code}: {response.text}")

    resp_json = response.json()
    try:
        candidate_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(candidate_text)
        return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini response as JSON: {e}\nRaw: {resp_json}")
