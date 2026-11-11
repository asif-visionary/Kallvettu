"""
Authentic Kalvettu (Stone Inscription) Sample Generator.
Creates realistic high-resolution stone inscription images with weathered granite/sandstone textures,
chiseled relief grooves, micro-cracks, and historical epigraphs.
"""

import os
from typing import Optional
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "samples")


def get_cross_platform_font_path() -> Optional[str]:
    """
    Discovers an available Tamil/Indic Unicode font across Windows, Linux, and macOS.
    Returns the path if found, or None to fall back to the default system font.
    """
    candidate_paths = [
        # Windows
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/NirmalaB.ttc",
        "C:/Windows/Fonts/Latha.ttf",
        "C:/Windows/Fonts/Vijaya.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansTamil-Regular.otf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/InaiMathi.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def generate_stone_texture(width=900, height=520, stone_type="granite"):
    """
    Synthesizes authentic stone surfaces with multi-scale roughness and mineral grain.
    """
    np.random.seed(42 if stone_type == "granite" else 84)

    # Base RGB tone
    if stone_type == "granite":
        # Dark Chola granite (Thanjavur temple style)
        base_color = np.array([72, 68, 65], dtype=np.float32)
        grain_scale = 22.0
    elif stone_type == "sandstone":
        # Warm cave rock (Mangulam / Pugalur style)
        base_color = np.array([140, 115, 90], dtype=np.float32)
        grain_scale = 28.0
    elif stone_type == "hero_stone":
        # Weathered schist / grey basalt (Nadukal style)
        base_color = np.array([95, 90, 85], dtype=np.float32)
        grain_scale = 25.0
    else:
        base_color = np.array([110, 105, 100], dtype=np.float32)
        grain_scale = 20.0

    # Multi-frequency noise
    # Fine grain
    fine_noise = np.random.normal(0, grain_scale, (height, width, 3))

    # Medium mineral patches via PIL resize for exact dimension match
    med_small = np.random.normal(0, grain_scale * 1.5, (height // 4, width // 4, 3))
    med_img = Image.fromarray(np.clip(med_small + 128, 0, 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    med_noise = np.array(med_img, dtype=np.float32) - 128.0

    # Broad geological shading
    coarse_small = np.random.normal(0, grain_scale * 2.0, (height // 16, width // 16, 3))
    coarse_img = Image.fromarray(np.clip(coarse_small + 128, 0, 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    coarse_noise = np.array(coarse_img, dtype=np.float32) - 128.0

    stone = base_color + fine_noise * 0.5 + med_noise * 0.35 + coarse_noise * 0.25
    stone = np.clip(stone, 15, 245).astype(np.uint8)
    return Image.fromarray(stone, mode="RGB")


def engrave_inscription_on_stone(stone_img, lines, font_size=32, stone_type="granite", header_title=""):
    """
    Engraves inscriptional text with chiseled depth, groove shadow, and rim lighting.
    """
    w, h = stone_img.size
    mask = Image.new("L", (w, h), 0)
    draw_mask = ImageDraw.Draw(mask)

    font_path = get_cross_platform_font_path()
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            small_font = ImageFont.truetype(font_path, 18)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw header tag
    if header_title:
        draw_mask.text((40, 30), f"❖ {header_title} ❖", font=small_font, fill=210)

    # Calculate line positioning
    y_start = 85
    line_spacing = int(font_size * 1.55)

    for i, line in enumerate(lines):
        y_pos = y_start + i * line_spacing
        draw_mask.text((50, y_pos), line, font=font, fill=255)

    # Convert mask to numpy for morphological chisel beveling
    mask_arr = np.array(mask, dtype=np.float32) / 255.0

    # Chisel groove shadow (displaced down-right)
    groove_shadow = ndi.shift(mask_arr, [2.5, 2.0], mode="constant", cval=0)
    # Chisel sunlight catch / bevel highlight (displaced up-left)
    rim_highlight = ndi.shift(mask_arr, [-1.5, -1.2], mode="constant", cval=0)

    stone_arr = np.array(stone_img, dtype=np.float32)

    # Darken carved grooves
    shadow_depth = 0.55 if stone_type == "granite" else 0.48
    for c in range(3):
        # Engrave sunken area
        stone_arr[:, :, c] = stone_arr[:, :, c] * (1.0 - mask_arr * shadow_depth)
        # Deep shadow in inner groove
        stone_arr[:, :, c] = stone_arr[:, :, c] * (1.0 - groove_shadow * 0.25)
        # Edge highlight catching light
        stone_arr[:, :, c] = np.clip(stone_arr[:, :, c] + rim_highlight * 45.0, 0, 255)

    # Add subtle rock fissure / crack running across
    y_crack = np.linspace(h * 0.2, h * 0.85, w) + np.sin(np.linspace(0, 15, w)) * 12
    for x in range(w):
        yc = int(y_crack[x])
        if 0 <= yc < h:
            stone_arr[max(0, yc-1):min(h, yc+2), x, :] *= 0.65

    return Image.fromarray(np.clip(stone_arr, 0, 255).astype(np.uint8), mode="RGB")


def create_all_samples():
    """Generates 6 authentic Kalvettu sample images."""
    os.makedirs(ASSETS_DIR, exist_ok=True)

    samples = [
        {
            "filename": "mangulam.jpg",
            "stone_type": "sandstone",
            "header": "மாங்குளம் குகைக் கல்வெட்டு - தமிழ் பிராமி (கி.மு. 3 ஆம் நூற்றாண்டு)",
            "font_size": 34,
            "lines": [
                "கணிய் நந்தாஸிரிய் இகுவன்",
                "வெள்ளறை நிகமத்தோர் கொடிபித பாளிய்",
                "பாண்டியன் நெடுஞ்செழியன் பணிகொள"
            ]
        },
        {
            "filename": "thanjavur.jpg",
            "stone_type": "granite",
            "header": "தஞ்சைப் பெருவுடையார் கோயில் - இராஜராஜ சோழன் கல்வெட்டு (கி.பி. 1010)",
            "font_size": 32,
            "lines": [
                "நாம் குடுத்தனவும் நம் அக்கன் குடுத்தனவும்",
                "நம் பெண்டுகள் குடுத்தனவும் மற்றார் குடுத்தனவும்",
                "இத்திருக் கற்றளியிலே கல்லிலே வெட்டுக என்று",
                "திருவாய் மொழிந்தருள..."
            ]
        },
        {
            "filename": "pugalur.jpg",
            "stone_type": "sandstone",
            "header": "புகழூர் ஆறுநாட்டார் மலை - சேரர் பிராமி கல்வெட்டு (கி.பி. 1-2 ஆம் நூற்றாண்டு)",
            "font_size": 33,
            "lines": [
                "யாற்றூர் செங்காயபன் உறை",
                "கொ ஆதன் செல்லிரும்பொறை மகன்",
                "பெருங்கடுங்கோ மகன் இளங்கடுங்கோ",
                "இளங்கோ ஆக அறுத்த கல்"
            ]
        },
        {
            "filename": "pulankurichi.jpg",
            "stone_type": "sandstone",
            "header": "பூலாங்குறிச்சி வட்டெழுத்துக் கல்வெட்டு (கி.பி. 5 ஆம் நூற்றாண்டு)",
            "font_size": 32,
            "lines": [
                "கோச்சேந்தன் கூற்றற்கு யாண்டு இருபத்தேழாவது",
                "கடிகைத்தாவளத்து பள்ளிச்சந்தமும்",
                "பிரம்மதேயமும் எல்லை வகுத்து அமைத்த ஆவணம்"
            ]
        },
        {
            "filename": "nadukal.jpg",
            "stone_type": "hero_stone",
            "header": "செங்கம் வீரக்கல் - ஆநிரை மீட்ட நடுகல் (கி.பி. 8 ஆம் நூற்றாண்டு)",
            "font_size": 33,
            "lines": [
                "தொறுக்கொண்டு போகிய பூசலில்",
                "நின்றெறிந்து பட்டான் சாத்தன்",
                "வீரனுக்கு நடப்பட்ட நடுகல்"
            ]
        },
        {
            "filename": "uttiramerur.jpg",
            "stone_type": "granite",
            "header": "உத்திரமேரூர் குடவோலை மக்களாட்சி முறை (கி.பி. 920 - பராந்தக சோழன்)",
            "font_size": 31,
            "lines": [
                "முப்பத்து குடும்பிலும் வாரியத்துக்கு ஓலை எழுதி",
                "குடத்திலே இட்டு சிறுவன் வழி எடுத்து...",
                "ஒழுக்கமும் கணக்குத் தூய்மையும் உடைய",
                "கள்ளமற்றார் வாரியம் செய்யக் கடவராகவும்"
            ]
        }
    ]

    print("Generating authentic stone inscription samples...")
    for s in samples:
        path = os.path.join(ASSETS_DIR, s["filename"])
        stone = generate_stone_texture(width=920, height=480, stone_type=s["stone_type"])
        inscribed = engrave_inscription_on_stone(
            stone,
            s["lines"],
            font_size=s["font_size"],
            stone_type=s["stone_type"],
            header_title=s["header"]
        )
        inscribed.save(path, quality=92)
        print(f"Generated: {s['filename']}")

    print("All sample inscriptions generated successfully!")


if __name__ == "__main__":
    create_all_samples()
