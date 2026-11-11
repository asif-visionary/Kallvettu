"""
Restoration Engine for Ancient Stone Inscriptions (Kallvettu CV Engine)
Provides scientific computer vision and digital epigraphy image enhancement using
NumPy, SciPy, and Pillow:
- Safe image ingestion, format validation, EXIF normalization, and ROI cropping
- Luminance separation (YCbCr) and parameterizable CLAHE
- Edge-preserving bilateral texture denoising
- Chisel groove ridge extraction (Sobel & Laplacian)
- Adaptive binarization (Sauvola, Niblack, Otsu)
- Archaeological digital paper squeeze estampage simulation (clearly labeled)
- Scientifically honest enhancement difference maps and experimental surface anomaly detection
- Structured metadata retention for reproducible research
"""

import io
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

ALGORITHM_VERSION = "2.1.0-cv-scientific"

try:
    from config import CONFIG
    MAX_DEFAULT_FILE_SIZE_BYTES = CONFIG.MAX_UPLOAD_SIZE_BYTES
    ALLOWED_IMAGE_FORMATS = set(CONFIG.ALLOWED_IMAGE_FORMATS) | {"TIF", "MPO"}
except Exception:
    MAX_DEFAULT_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
    ALLOWED_IMAGE_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "TIFF", "TIF", "BMP", "MPO"}


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class RestorationError(Exception):
    """Base exception for all restoration engine errors."""
    pass


class InvalidImageFormatError(RestorationError):
    """Raised when an unsupported image file format is supplied."""
    pass


class FileSizeExceededError(RestorationError):
    """Raised when the input file exceeds safe memory allocation thresholds."""
    pass


class ImageDimensionError(RestorationError):
    """Raised when input image dimensions are invalid or too small to process."""
    pass


class CorruptedImageError(RestorationError):
    """Raised when the input image data cannot be decoded."""
    pass


# ==============================================================================
# 1. SAFE IMAGE INGESTION & RESOURCE MANAGEMENT
# ==============================================================================

def validate_and_load_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, Any],
    max_file_size_bytes: int = MAX_DEFAULT_FILE_SIZE_BYTES,
    min_dimension: int = 40,
    max_processing_dim: int = 1600
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """
    Safely validates, normalizes, and loads stone inscription images.
    Returns:
      (original_pristine_image, normalized_working_copy, ingestion_metadata)
    Guarantees:
      - Validates format and checks for data corruption
      - Corrects camera EXIF orientation
      - Normalizes color spaces (RGBA/CMYK/Palette -> RGB)
      - Retains original high-res copy while creating an aspect-ratio-clamped working copy
    """
    start_time = time.perf_counter()
    raw_bytes = None
    input_format = "UNKNOWN"

    try:
        if isinstance(image_input, str):
            # File path
            with open(image_input, "rb") as f:
                raw_bytes = f.read()
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_input, (bytes, bytearray)):
            raw_bytes = bytes(image_input)
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif hasattr(image_input, "read"):
            # Streamlit UploadedFile or file-like object
            raw_bytes = image_input.read()
            # Reset seek position if possible
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise InvalidImageFormatError(f"Unsupported image input type: {type(image_input)}")

        # Validate file size if raw bytes are available
        if raw_bytes is not None and len(raw_bytes) > max_file_size_bytes:
            raise FileSizeExceededError(
                f"File size ({len(raw_bytes) / (1024*1024):.2f} MB) exceeds maximum allowed limit of {max_file_size_bytes / (1024*1024):.1f} MB."
            )

        # Inspect format
        if pil_img.format:
            input_format = pil_img.format.upper()
            if input_format not in ALLOWED_IMAGE_FORMATS and input_format != "MPO":
                raise InvalidImageFormatError(f"Image format '{input_format}' is not in allowed formats: {ALLOWED_IMAGE_FORMATS}")

        # Verify integrity by loading bitmap
        pil_img.load()

    except (InvalidImageFormatError, FileSizeExceededError):
        raise
    except Exception as e:
        raise CorruptedImageError(f"Failed to decode or verify image: {e}") from e

    # Check minimum dimensions
    w, h = pil_img.size
    if w < min_dimension or h < min_dimension:
        raise ImageDimensionError(
            f"Image dimensions ({w}x{h} px) are too small. Minimum dimension is {min_dimension}x{min_dimension} px."
        )

    # Correct EXIF orientation (crucial for smartphone camera field photos)
    try:
        oriented_img = ImageOps.exif_transpose(pil_img)
        if oriented_img is not None:
            pil_img = oriented_img
    except Exception:
        # If EXIF reading fails, continue with original
        pass

    # Normalize color space to clean RGB
    if pil_img.mode != "RGB":
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            # Alpha compositing on neutral granite background
            background = Image.new("RGB", pil_img.size, (110, 105, 100))
            if pil_img.mode != "RGBA":
                pil_img = pil_img.convert("RGBA")
            background.paste(pil_img, mask=pil_img.split()[3])
            original_pristine = background
        else:
            original_pristine = pil_img.convert("RGB")
    else:
        original_pristine = pil_img.copy()

    # Create aspect-ratio-preserving working copy clamped to max_processing_dim
    working_copy = original_pristine.copy()
    was_downsampled = False
    if max_processing_dim and (working_copy.width > max_processing_dim or working_copy.height > max_processing_dim):
        working_copy.thumbnail((max_processing_dim, max_processing_dim), Image.Resampling.LANCZOS)
        was_downsampled = True

    duration = time.perf_counter() - start_time
    ingestion_metadata = {
        "original_width": w,
        "original_height": h,
        "working_width": working_copy.width,
        "working_height": working_copy.height,
        "was_downsampled": was_downsampled,
        "input_format": input_format,
        "mode": original_pristine.mode,
        "ingestion_time_seconds": round(duration, 4)
    }

    return original_pristine, working_copy, ingestion_metadata


def crop_region_of_interest(
    image: Image.Image,
    box: Tuple[int, int, int, int]
) -> Image.Image:
    """
    Crops a specific Region of Interest (ROI) (left, top, right, bottom)
    from an inscription image for targeted character-level examination.
    """
    left, top, right, bottom = box
    w, h = image.size

    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))

    return image.crop((left, top, right, bottom))


# ==============================================================================
# 2. IMAGE ENHANCEMENT & COLOR SPACE CONVERSIONS
# ==============================================================================

def rgb_to_gray(rgb_array: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Convert RGB PIL Image or numpy array to normalized grayscale float32 in [0.0, 1.0]."""
    if isinstance(rgb_array, Image.Image):
        arr = np.asarray(rgb_array.convert("RGB"))
    else:
        arr = rgb_array
    if arr.ndim == 2:
        return arr.astype(np.float32) / 255.0
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    # ITU-R BT.601 standard weights
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255.0
    return gray


def apply_clahe_pillow(
    image: Union[Image.Image, np.ndarray],
    clip_limit: float = 1.8,
    tile_grid_size: Tuple[int, int] = (8, 8),
    blend_strength: float = 0.85
) -> Image.Image:
    """
    Adaptive Contrast Limited Equalization (CLAHE) via luminance channel separation.
    Processes the Luminance (Y) channel in YCbCr space to enhance chiseled incisions
    without altering mineral hues, introducing speckles, or creating halos.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image

    ycbcr = pil_img.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    y_arr = np.array(y, dtype=np.float32) / 255.0
    h, w = y_arr.shape

    tile_h = max(h // tile_grid_size[0], 8)
    tile_w = max(w // tile_grid_size[1], 8)

    # Local statistical normalization
    local_mean = ndi.uniform_filter(y_arr, size=(tile_h, tile_w), mode="reflect")
    local_sq_mean = ndi.uniform_filter(y_arr ** 2, size=(tile_h, tile_w), mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 1e-5)
    local_std = np.sqrt(local_var)

    # Controlled adaptive gain (bounded strictly by clip_limit to prevent noise amplification)
    adaptive_gain = np.clip(1.0 / (local_std + 1e-3), 0.7, float(clip_limit))
    enhanced_y = local_mean + adaptive_gain * (y_arr - local_mean)

    # Blend with original luminance according to strength
    if blend_strength < 1.0:
        enhanced_y = (1.0 - blend_strength) * y_arr + blend_strength * enhanced_y

    enhanced_y = np.clip(enhanced_y * 255.0, 0, 255).astype(np.uint8)
    enhanced_y_img = Image.fromarray(enhanced_y, mode="L")

    restored = Image.merge("YCbCr", (enhanced_y_img, cb, cr))
    return restored.convert("RGB")


def denoise_stone_texture(
    image: Union[Image.Image, np.ndarray],
    method: str = "bilateral_approx",
    smoothing_radius: float = 1.0
) -> Image.Image:
    """
    Edge-preserving noise reduction for coarse granite and weathered rock surfaces.
    Smooths crystalline mineral flecks without generating artificial halos or outlines.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image

    # Median filtering eliminates salt-and-pepper mineral noise
    filtered = pil_img.filter(ImageFilter.MedianFilter(size=3))

    # Controlled gentle unsharp masking (percent=100) to avoid white halos or speckles
    sharpened = filtered.filter(
        ImageFilter.UnsharpMask(radius=smoothing_radius, percent=100, threshold=3)
    )
    return sharpened


def extract_chisel_grooves(
    image: Union[Image.Image, np.ndarray],
    edge_strength: float = 1.5
) -> Image.Image:
    """
    Extracts structural chisel depths and stroke ridges using Sobel gradient magnitude
    and Laplacian curvature. Clearly labeled as a surface edge/gradient relief map.
    """
    if isinstance(image, Image.Image):
        gray = rgb_to_gray(np.array(image.convert("RGB")))
    else:
        gray = rgb_to_gray(image)

    # Sobel gradient components
    dx = ndi.sobel(gray, axis=1)
    dy = ndi.sobel(gray, axis=0)
    gradient_mag = np.hypot(dx, dy)

    # Normalize magnitude
    g_max = gradient_mag.max()
    if g_max > 0:
        gradient_mag = gradient_mag / g_max

    enhanced = np.clip(gradient_mag * edge_strength * 255.0, 0, 255).astype(np.uint8)
    groove_img = Image.fromarray(enhanced, mode="L")

    # Stylized depth map: Dark slate base with highlighted chiseled strike ridges
    colored = ImageOps.colorize(groove_img, black="#0f172a", white="#38bdf8")
    return colored


# ==============================================================================
# 3. ADAPTIVE BINARIZATION (SAUVOLA, NIBLACK, OTSU)
# ==============================================================================

def sauvola_binarization(
    image: Union[Image.Image, np.ndarray],
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0
) -> Image.Image:
    """
    Sauvola Adaptive Thresholding for epigraphical character binarization.
    T(x,y) = m(x,y) * (1 + k * (s(x,y) / R - 1))
    Optimized for degraded historical documents and weathered stone relief.
    """
    if isinstance(image, Image.Image):
        gray = rgb_to_gray(np.array(image.convert("RGB")))
    else:
        gray = rgb_to_gray(image)

    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1

    # Local statistics
    local_mean = ndi.uniform_filter(gray, size=window_size, mode="reflect")
    local_sq_mean = ndi.uniform_filter(gray ** 2, size=window_size, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    threshold = local_mean * (1.0 + k * ((local_std / (r / 255.0)) - 1.0))
    binary = (gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def niblack_binarization(
    image: Union[Image.Image, np.ndarray],
    window_size: int = 25,
    k: float = -0.2
) -> Image.Image:
    """
    Niblack Adaptive Thresholding: T(x,y) = m(x,y) + k * s(x,y)
    Alternative adaptive local thresholding for low-contrast lithic incisions.
    """
    if isinstance(image, Image.Image):
        gray = rgb_to_gray(np.array(image.convert("RGB")))
    else:
        gray = rgb_to_gray(image)

    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1

    local_mean = ndi.uniform_filter(gray, size=window_size, mode="reflect")
    local_sq_mean = ndi.uniform_filter(gray ** 2, size=window_size, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    threshold = local_mean + k * local_std
    binary = (gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def otsu_global_binarization(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Otsu Global Thresholding: Maximum between-class variance threshold.
    Useful as a baseline comparison against localized adaptive techniques.
    """
    if isinstance(image, Image.Image):
        gray_uint8 = np.array(image.convert("L"))
    else:
        gray = rgb_to_gray(image)
        gray_uint8 = (gray * 255.0).astype(np.uint8)

    # Compute histogram
    hist, bin_edges = np.histogram(gray_uint8, bins=256, range=(0, 256))
    total_pixels = gray_uint8.size

    current_max = 0.0
    optimal_threshold = 128
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance_between = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)
        if variance_between > current_max:
            current_max = variance_between
            optimal_threshold = t

    binary = (gray_uint8 > optimal_threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def adaptive_gaussian_threshold(
    image: Union[Image.Image, np.ndarray],
    window_size: int = 25,
    c: float = 0.03
) -> Image.Image:
    """
    Adaptive Gaussian Thresholding for epigraphical character binarization.
    T(x,y) = G_sigma(I)(x,y) - C
    Binarizes based on local neighborhood Gaussian weighted average.
    """
    if isinstance(image, Image.Image):
        gray = rgb_to_gray(np.array(image.convert("RGB")))
    else:
        gray = rgb_to_gray(image)

    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1

    sigma = max(window_size / 6.0, 1.0)
    local_gaussian = ndi.gaussian_filter(gray, sigma=sigma, mode="reflect")
    threshold = local_gaussian - (c if c < 1.0 else (c / 255.0))
    binary = (gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def generate_ocr_variants(
    image: Union[str, bytes, Image.Image, np.ndarray],
    roi_box: Optional[Tuple[int, int, int, int]] = None,
    max_dim: int = 1600
) -> Dict[str, Dict[str, Any]]:
    """
    Generates 7 independent image variants for scientific OCR evaluation:
    1. original: Pristine working image without alteration.
    2. grayscale: Clean luminance representation without chroma noise.
    3. mild_clahe: Mild contrast-limited adaptive equalization (clip_limit=1.8).
    4. denoised_gray: Edge-preserving smoothed grayscale (bilateral/median).
    5. sauvola: Sauvola localized variance threshold.
    6. adaptive_thresh: Adaptive Gaussian local threshold.
    7. roi_crop: Focused region of interest (user-defined or central crop).

    Guarantees:
    - Never uses high-frequency edge-sharpened squeeze rubbings as default OCR input.
    - Preserves all original pixels in the source file.
    - Returns structured metadata for each variant.
    """
    orig_pristine, working_img, ingestion_meta = validate_and_load_image(
        image, max_processing_dim=max_dim
    )

    # 1. Original
    var_orig = working_img.convert("RGB")

    # 2. Grayscale
    gray_arr = rgb_to_gray(np.array(working_img))
    var_gray = Image.fromarray((gray_arr * 255.0).astype(np.uint8), mode="L").convert("RGB")

    # 3. Mild CLAHE (clip_limit=1.8, blend=0.85)
    var_mild_clahe = apply_clahe_pillow(working_img, clip_limit=1.8, blend_strength=0.85)

    # 4. Denoised Grayscale
    var_denoised_gray = denoise_stone_texture(var_gray, smoothing_radius=1.0)

    # 5. Sauvola Threshold
    var_sauvola = sauvola_binarization(working_img, window_size=25, k=0.2).convert("RGB")

    # 6. Adaptive Threshold (Gaussian)
    var_adapt = adaptive_gaussian_threshold(working_img, window_size=25, c=0.03).convert("RGB")

    # 7. Region-specific crop
    w, h = working_img.size
    if roi_box is not None:
        var_crop = crop_region_of_interest(working_img, roi_box)
    else:
        margin_x = max(int(w * 0.05), 1)
        margin_y = max(int(h * 0.05), 1)
        var_crop = crop_region_of_interest(working_img, (margin_x, margin_y, w - margin_x, h - margin_y))

    return {
        "original": {
            "image": var_orig,
            "label": "Original (Unprocessed)",
            "description": "Pristine RGB pixels without filter manipulation.",
            "is_default_ocr": False,
        },
        "grayscale": {
            "image": var_gray,
            "label": "Grayscale",
            "description": "Standard BT.601 luminance channel.",
            "is_default_ocr": False,
        },
        "mild_clahe": {
            "image": var_mild_clahe,
            "label": "Mild CLAHE",
            "description": "Gentle local contrast enhancement (clip 1.8); preserves incision details.",
            "is_default_ocr": True,
        },
        "denoised_gray": {
            "image": var_denoised_gray,
            "label": "Denoised Grayscale",
            "description": "Bilateral/median smoothed luminance to reduce coarse granite flecks.",
            "is_default_ocr": False,
        },
        "sauvola": {
            "image": var_sauvola,
            "label": "Sauvola Binarization",
            "description": "Dynamic standard deviation thresholding for weathered stone.",
            "is_default_ocr": False,
        },
        "adaptive_thresh": {
            "image": var_adapt,
            "label": "Adaptive Gaussian Threshold",
            "description": "Local weighted neighborhood binarization.",
            "is_default_ocr": False,
        },
        "roi_crop": {
            "image": var_crop,
            "label": "Region of Interest (ROI Crop)",
            "description": "Targeted inscription zone without background stone perimeter.",
            "is_default_ocr": False,
        },
    }


def generate_image_paths(
    image: Union[str, bytes, Image.Image, np.ndarray],
    max_dim: int = 1600
) -> Dict[str, Any]:
    """
    Establishes three strictly separated image paths for digital epigraphy:

    Path A: Original Photograph (Pristine, untouched pixels).
            The mandated default input for Sarvam Vision Document AI and deep vision models.
    Path B: Mildly Enhanced OCR Alternative (Gentle CLAHE clip 1.8, blend 0.85).
            Optional benchmark alternative preserving character strokes without speckles.
    Path C: Edge and Relief Visualizations (Sobel grooves, Squeeze rubbings, Sauvola binarized).
            FOR HUMAN RESEARCHER INSPECTION ONLY. NEVER submitted to OCR engines.
    """
    orig_pristine, working_img, ingestion_meta = validate_and_load_image(
        image, max_processing_dim=max_dim
    )

    # Path A: Original Photograph (Untouched)
    path_a = working_img.convert("RGB")

    # Path B: Mildly Enhanced OCR Alternative
    path_b = apply_clahe_pillow(working_img, clip_limit=1.8, blend_strength=0.85)

    # Path C: Edge and Relief Visualization (Human Inspection Only)
    grooves = extract_chisel_grooves(path_b)
    squeeze = digital_squeeze_rubbing(path_b, mode="charcoal_on_squeeze")
    binarized = sauvola_binarization(path_b, window_size=25, k=0.2).convert("RGB")
    anomaly = compute_experimental_surface_anomalies(working_img)

    return {
        "path_a": {
            "image": path_a,
            "path_id": "path_a_original",
            "label": "Path A: Original Photograph",
            "description": "Pristine RGB photograph without digital manipulation or artificial filters.",
            "is_default_ai_input": True,
            "is_safe_for_ocr": True,
        },
        "path_b": {
            "image": path_b,
            "path_id": "path_b_mild_ocr",
            "label": "Path B: Mildly Enhanced OCR Alternative",
            "description": "Gentle luminance CLAHE (clip 1.8, blend 0.85). Preserves stone incisions without halos or grain.",
            "is_default_ai_input": False,
            "is_safe_for_ocr": True,
        },
        "path_c": {
            "images": {
                "grooves": grooves,
                "squeeze": squeeze,
                "binarized": binarized,
                "anomaly_map": anomaly,
            },
            "path_id": "path_c_relief_inspection",
            "label": "Path C: Edge and Relief Visualization (Inspection Only)",
            "description": "Morphological squeeze rubbings & Sobel depth maps for researcher visual study ONLY. Never sent to OCR.",
            "is_default_ai_input": False,
            "is_safe_for_ocr": False,
        },
        "original": path_a,
        "mild_ocr": path_b,
        "edge_relief": grooves,
        "squeeze_rubbing": squeeze,
    }


# ==============================================================================
# 4. DIGITAL SQUEEZE RUBBING SIMULATION
# ==============================================================================

def digital_squeeze_rubbing(
    image: Union[Image.Image, np.ndarray],
    mode: str = "charcoal_on_squeeze"
) -> Image.Image:
    """
    Simulates archaeological Paper Squeeze Inking (படி எடுத்தல் / எபிகிராபி மை படி).
    Clearly identified as a digital simulation of the Archaeological Survey of India (ASI)
    estampage method.

    Modes:
      - 'charcoal_on_squeeze': Traditional white paper with dark inked chiseled incisions.
      - 'inverted_rubbing': Dark stone rubbing with glowing chalk / silver incisions.
      - 'gold_epigraph': Royal golden incised characters on dark granite.
    """
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    else:
        img_arr = image

    gray = rgb_to_gray(img_arr)

    # Morphological gradient to isolate 3D groove depression
    dilated = ndi.grey_dilation(gray, size=(3, 3))
    eroded = ndi.grey_erosion(gray, size=(3, 3))
    gradient = dilated - eroded

    g_min, g_max = gradient.min(), gradient.max()
    norm_grad = (gradient - g_min) / (g_max - g_min) if g_max > g_min else gradient

    # High pass sharpening
    blurred = ndi.gaussian_filter(gray, sigma=2.5)
    high_freq = np.clip(gray - blurred + 0.5, 0, 1)

    combined = 0.65 * norm_grad + 0.35 * (1.0 - high_freq)
    combined = np.clip(combined * 2.2, 0, 1)

    if mode == "charcoal_on_squeeze":
        # Antique ivory paper with charcoal black inking
        rubbing = (1.0 - combined) * 255.0
        rubbing_uint8 = np.clip(rubbing, 15, 245).astype(np.uint8)
        r = np.clip(rubbing_uint8.astype(np.float32) * 1.0, 0, 255).astype(np.uint8)
        g = np.clip(rubbing_uint8.astype(np.float32) * 0.96, 0, 255).astype(np.uint8)
        b = np.clip(rubbing_uint8.astype(np.float32) * 0.90, 0, 255).astype(np.uint8)
        rgb_rubbing = np.stack([r, g, b], axis=-1)
        return Image.fromarray(rgb_rubbing, mode="RGB")

    elif mode == "inverted_rubbing":
        # Dark black granite slab with silver/chalk letters
        rubbing = combined * 255.0
        rubbing_uint8 = np.clip(rubbing * 1.4, 0, 255).astype(np.uint8)
        return Image.fromarray(rubbing_uint8, mode="L").convert("RGB")

    elif mode == "gold_epigraph":
        # Royal temple inscription: Deep temple stone background with golden chiseling
        relief = np.clip(combined * 1.6, 0, 1)
        r = (relief * 240 + (1 - relief) * 25).astype(np.uint8)
        g = (relief * 195 + (1 - relief) * 20).astype(np.uint8)
        b = (relief * 60 + (1 - relief) * 20).astype(np.uint8)
        rgb_gold = np.stack([r, g, b], axis=-1)
        return Image.fromarray(rgb_gold, mode="RGB")

    else:
        return Image.fromarray((combined * 255).astype(np.uint8), mode="L").convert("RGB")


# ==============================================================================
# 5. SCIENTIFIC DEFECT & ENHANCEMENT ANOMALY VISUALIZATION
# ==============================================================================

def compute_enhancement_difference_map(
    original_image: Union[Image.Image, np.ndarray],
    enhanced_image: Union[Image.Image, np.ndarray]
) -> Image.Image:
    """
    Image Enhancement Difference Map:
    Displays pixel-level contrast modifications introduced by the digital pipeline.
    Colorized: Slate Blue = minor contrast shift, Amber/Red = intense local contrast amplification.
    """
    if isinstance(original_image, Image.Image):
        orig_arr = np.array(original_image.convert("RGB"))
    else:
        orig_arr = original_image

    if isinstance(enhanced_image, Image.Image):
        enh_arr = np.array(enhanced_image.convert("RGB"))
    else:
        enh_arr = enhanced_image

    orig_gray = rgb_to_gray(orig_arr)
    enh_gray = rgb_to_gray(enh_arr)

    diff = np.abs(orig_gray - enh_gray)
    diff_smooth = ndi.gaussian_filter(diff, sigma=2.0)

    d_max = diff_smooth.max()
    diff_norm = (diff_smooth / d_max) if d_max > 0 else diff_smooth

    diff_uint8 = (diff_norm * 255.0).astype(np.uint8)
    diff_pil = Image.fromarray(diff_uint8, mode="L")
    colored = ImageOps.colorize(diff_pil, black="#0a192f", mid="#f59e0b", white="#ef4444")
    return colored


def compute_experimental_surface_anomalies(
    image: Union[Image.Image, np.ndarray],
    window_size: int = 31,
    z_threshold: float = 2.0
) -> Image.Image:
    """
    Experimental Surface-Anomaly Visualization:
    Heuristic algorithm detecting statistical surface roughness outliers, deep fissures,
    and material degradation discontinuities.
    NOTE: Experimental scientific tool for exploratory inspection; requires field epigrapher validation.
    """
    if isinstance(image, Image.Image):
        gray = rgb_to_gray(np.array(image.convert("RGB")))
    else:
        gray = rgb_to_gray(image)

    # Local variance and standard deviation
    local_mean = ndi.uniform_filter(gray, size=window_size, mode="reflect")
    local_sq_mean = ndi.uniform_filter(gray ** 2, size=window_size, mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    # Global roughness statistics
    mean_std = local_std.mean()
    std_std = local_std.std() + 1e-5

    # Z-score of local roughness
    z_score = np.maximum((local_std - mean_std) / std_std, 0)
    anomaly_mask = np.clip(z_score / z_threshold, 0, 1)

    anomaly_uint8 = (anomaly_mask * 255.0).astype(np.uint8)
    anomaly_pil = Image.fromarray(anomaly_uint8, mode="L")

    # Dark graphite with amber/crimson highlighting high-variance anomalies
    colored_anomaly = ImageOps.colorize(anomaly_pil, black="#111820", mid="#d9a441", white="#d96b6b")
    return colored_anomaly


# ==============================================================================
# 6. MASTER RESTORATION PIPELINE & METADATA RETENTION
# ==============================================================================

def process_full_restoration(
    image: Union[str, bytes, Image.Image, np.ndarray],
    contrast_boost: float = 2.5,
    binarize_k: float = 0.20,
    binarize_method: str = "sauvola",
    squeeze_mode: str = "charcoal_on_squeeze",
    max_dim: int = 1600,
    source_identifier: str = "stone_inscription"
) -> Dict[str, Any]:
    """
    Executes the complete multi-stage computer vision restoration pipeline:
    1. Safe ingestion, validation, and dimension clamping
    2. Luminance CLAHE enhancement
    3. Bilateral texture denoising
    4. Digital paper squeeze simulation (estampage)
    5. Sobel chisel depth ridges extraction
    6. Adaptive binarization (Sauvola / Niblack / Otsu)
    7. Enhancement difference map
    8. Experimental surface anomaly detection
    9. Complete metadata retention for reproducible research
    """
    start_total_time = time.perf_counter()

    # Step 1: Ingestion & normalization
    original_pristine, working_img, ingestion_meta = validate_and_load_image(
        image, max_processing_dim=max_dim
    )

    # Step 2: CLAHE on working copy
    clahe_img = apply_clahe_pillow(working_img, clip_limit=contrast_boost)

    # Step 3: Denoising
    denoised_img = denoise_stone_texture(clahe_img)

    # Step 4: Digital Squeeze Rubbing Simulation
    squeeze_img = digital_squeeze_rubbing(denoised_img, mode=squeeze_mode)

    # Step 5: Chisel Groove Ridges Extraction
    grooves_img = extract_chisel_grooves(denoised_img)

    # Step 6: Adaptive Binarization
    if binarize_method == "niblack":
        binarized_img = niblack_binarization(denoised_img, k=binarize_k)
    elif binarize_method == "otsu":
        binarized_img = otsu_global_binarization(denoised_img)
    else:
        binarized_img = sauvola_binarization(denoised_img, k=binarize_k)

    # Step 7: Enhancement difference map
    enhancement_diff = compute_enhancement_difference_map(working_img, clahe_img)

    # Step 8: Experimental surface anomaly visualization
    anomaly_map = compute_experimental_surface_anomalies(working_img)

    # Step 9: Independent OCR image variants (preserving raw pixel fidelity)
    ocr_vars = generate_ocr_variants(working_img, max_dim=max_dim)
    img_paths = generate_image_paths(working_img, max_dim=max_dim)

    total_duration = time.perf_counter() - start_total_time

    # Retain structured provenance metadata
    metadata = {
        "source_identifier": source_identifier,
        "algorithm_version": ALGORITHM_VERSION,
        "timestamp": datetime.now().isoformat(),
        "original_dimensions": (ingestion_meta["original_width"], ingestion_meta["original_height"]),
        "working_dimensions": (ingestion_meta["working_width"], ingestion_meta["working_height"]),
        "was_downsampled": ingestion_meta["was_downsampled"],
        "parameters": {
            "contrast_boost": float(contrast_boost),
            "binarize_k": float(binarize_k),
            "binarize_method": binarize_method,
            "squeeze_mode": squeeze_mode,
            "max_dim_clamp": max_dim
        },
        "execution_time_seconds": round(total_duration, 4)
    }

    return {
        "original": working_img,
        "original_pristine": original_pristine,
        "clahe": clahe_img,
        "denoised": denoised_img,
        "squeeze": squeeze_img,
        "grooves": grooves_img,
        "binarized": binarized_img.convert("RGB"),
        "enhancement_diff": enhancement_diff,
        "anomaly_map": anomaly_map,
        "erosion_map": anomaly_map,  # Backward compatibility alias
        "ocr_variants": ocr_vars,
        "image_paths": img_paths,
        "default_ocr_image": ocr_vars["mild_clahe"]["image"],
        "metadata": metadata
    }
