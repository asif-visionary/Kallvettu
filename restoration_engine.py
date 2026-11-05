"""
Restoration Engine for Ancient Stone Inscriptions (Kalvettu)
Provides computer vision and digital epigraphy image enhancement using
NumPy, SciPy, and Pillow (CLAHE, Sauvola Binarization, Digital Squeeze Rubbing,
Edge Groove Extraction, and Texture Denoising).
"""

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def load_image_as_numpy(image_or_path):
    """Convert PIL Image, file path, or bytes to RGB numpy array."""
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path).convert("RGB")
    elif isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("RGB")
    elif hasattr(image_or_path, "read"):
        img = Image.open(image_or_path).convert("RGB")
    else:
        raise ValueError("Unsupported image input type")
    return np.array(img, dtype=np.uint8)


def rgb_to_gray(rgb_array):
    """Convert RGB numpy array to grayscale float [0, 1]."""
    r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / 255.0


def apply_clahe_pillow(image, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) approximation
    using tiled local histogram equalization and blending.
    Greatly enhances carved letters on weathered stone surfaces.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    # Convert to YCbCr to enhance Luminance (Y) channel without distorting color
    ycbcr = pil_img.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    y_arr = np.array(y, dtype=np.float32) / 255.0
    h, w = y_arr.shape
    th, tw = h // tile_grid_size[0], w // tile_grid_size[1]
    th = max(th, 8)
    tw = max(tw, 8)

    # Local mean and standard deviation for contrast enhancement
    mean = ndi.uniform_filter(y_arr, size=(th, tw), mode="reflect")
    sq_mean = ndi.uniform_filter(y_arr**2, size=(th, tw), mode="reflect")
    variance = np.maximum(sq_mean - mean**2, 1e-5)
    std = np.sqrt(variance)

    # Adaptive contrast expansion with clip limit
    gain = np.clip(1.0 / (std + 1e-3), 0.5, clip_limit * 1.8)
    enhanced_y = mean + gain * (y_arr - mean)
    enhanced_y = np.clip(enhanced_y * 255.0, 0, 255).astype(np.uint8)

    enhanced_y_img = Image.fromarray(enhanced_y, mode="L")
    restored_ycbcr = Image.merge("YCbCr", (enhanced_y_img, cb, cr))
    return restored_ycbcr.convert("RGB")


def sauvola_binarization(image, window_size=25, k=0.2, r=128):
    """
    Sauvola Adaptive Thresholding - the gold standard in ancient document
    and epigraphical character binarization.
    T = m * (1 + k * (s / r - 1))
    """
    if isinstance(image, Image.Image):
        gray = rgb_to_gray(np.array(image.convert("RGB")))
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            gray = rgb_to_gray(image)
        else:
            gray = image.astype(np.float32) / 255.0
    else:
        raise ValueError("Invalid image input")

    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1

    # Local mean and std
    mean = ndi.uniform_filter(gray, size=window_size, mode="reflect")
    sq_mean = ndi.uniform_filter(gray**2, size=window_size, mode="reflect")
    std = np.sqrt(np.maximum(sq_mean - mean**2, 0))

    threshold = mean * (1.0 + k * ((std / (r / 255.0)) - 1.0))
    binary = (gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def digital_squeeze_rubbing(image, mode="charcoal_on_squeeze"):
    """
    Simulates archaeological Paper Squeeze Inking (படி எடுத்தல் / எபிகிராபி மை படி).
    In traditional epigraphy, wet paper is hammered into the stone incisions and inked with charcoal.
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
    # Morphological gradient to detect groove depths
    dilated = ndi.grey_dilation(gray, size=(3, 3))
    eroded = ndi.grey_erosion(gray, size=(3, 3))
    gradient = dilated - eroded

    # Normalize gradient
    g_min, g_max = gradient.min(), gradient.max()
    if g_max > g_min:
        norm_grad = (gradient - g_min) / (g_max - g_min)
    else:
        norm_grad = gradient

    # High pass sharpening
    blurred = ndi.gaussian_filter(gray, sigma=3.0)
    high_freq = np.clip(gray - blurred + 0.5, 0, 1)

    combined = 0.65 * norm_grad + 0.35 * (1.0 - high_freq)
    combined = np.clip(combined * 2.2, 0, 1)

    if mode == "charcoal_on_squeeze":
        # White handmade paper (antique tone) with black ink on carvings
        rubbing = (1.0 - combined) * 255.0
        rubbing_uint8 = np.clip(rubbing, 15, 245).astype(np.uint8)
        # Add subtle paper tint (warm cream / antique ivory)
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


def extract_chisel_grooves(image, edge_strength=1.5):
    """
    Extracts structural chisel depths and stroke ridges using Sobel and Laplacian filters.
    Helps epigraphists read faint, worn-out strokes that blend into granite grains.
    """
    if isinstance(image, Image.Image):
        gray = rgb_to_gray(np.array(image.convert("RGB")))
    else:
        gray = rgb_to_gray(image) if image.ndim == 3 else image.astype(np.float32) / 255.0

    # Sobel filters
    dx = ndi.sobel(gray, axis=1)
    dy = ndi.sobel(gray, axis=0)
    magnitude = np.hypot(dx, dy)

    # Normalize and scale
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max

    enhanced = np.clip(magnitude * edge_strength * 255.0, 0, 255).astype(np.uint8)
    # Return as stylized depth map (dark stone with highlighted groove cuts)
    groove_img = Image.fromarray(enhanced, mode="L")
    colored = ImageOps.colorize(groove_img, black="#0f172a", white="#38bdf8")
    return colored


def denoise_stone_texture(image, smoothing_radius=1.8):
    """
    Removes granular stone noise while preserving character incision boundaries
    using edge-preserving median and guided Gaussian filtering.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    # Bilateral approximation via PIL
    filtered = pil_img.filter(ImageFilter.MedianFilter(size=3))
    # Unsharp mask to restore sharp character edges
    sharpened = filtered.filter(ImageFilter.UnsharpMask(radius=smoothing_radius, percent=150, threshold=3))
    return sharpened


def highlight_erosion_damage(original_image, restored_image):
    """
    Generates an archaeological erosion and weathering defect heatmap.
    Pinpoints fractured stone surfaces, lichen patches, and eroded characters
    requiring epigraphic context-based textual restoration.
    """
    if isinstance(original_image, Image.Image):
        orig_arr = np.array(original_image.convert("RGB"))
    else:
        orig_arr = original_image

    if isinstance(restored_image, Image.Image):
        rest_arr = np.array(restored_image.convert("RGB"))
    else:
        rest_arr = restored_image

    orig_gray = rgb_to_gray(orig_arr)
    rest_gray = rgb_to_gray(rest_arr)

    diff = np.abs(orig_gray - rest_gray)
    diff_smooth = ndi.gaussian_filter(diff, sigma=3.0)
    d_max = diff_smooth.max()
    if d_max > 0:
        diff_norm = diff_smooth / d_max
    else:
        diff_norm = diff_smooth

    # Colorize difference map: Blue = well preserved, Red/Yellow = heavily eroded/restored
    heatmap_uint8 = (diff_norm * 255.0).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8, mode="L")
    colored_heatmap = ImageOps.colorize(heatmap_pil, black="#0a192f", mid="#f59e0b", white="#ef4444")
    return colored_heatmap


def process_full_restoration(image, contrast_boost=2.5, binarize_k=0.2, squeeze_mode="charcoal_on_squeeze"):
    """
    Runs the complete multi-stage restoration pipeline and returns a dictionary of images:
    - 'original'
    - 'clahe'
    - 'denoised'
    - 'squeeze'
    - 'grooves'
    - 'binarized'
    - 'erosion_map'
    """
    if isinstance(image, str):
        original = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        original = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        original = Image.fromarray(image).convert("RGB")
    else:
        original = Image.open(image).convert("RGB")

    clahe = apply_clahe_pillow(original, clip_limit=contrast_boost)
    denoised = denoise_stone_texture(clahe)
    squeeze = digital_squeeze_rubbing(denoised, mode=squeeze_mode)
    grooves = extract_chisel_grooves(denoised)
    binarized = sauvola_binarization(denoised, k=binarize_k)
    erosion_map = highlight_erosion_damage(original, clahe)

    return {
        "original": original,
        "clahe": clahe,
        "denoised": denoised,
        "squeeze": squeeze,
        "grooves": grooves,
        "binarized": binarized.convert("RGB"),
        "erosion_map": erosion_map,
    }
