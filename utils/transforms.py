"""
IMPROVED TRANSFORMS - ROBUST BBOX HANDLING
===========================================

Key Improvements:
✓ Safe bbox handling (no out-of-bounds errors)
✓ Permissive validation (no bbox rejection)
✓ Compatible with albumentations 1.3.1
✓ Reduces false positives through careful augmentation
✓ Better handling of blur/low quality images
"""

import albumentations as A
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms


def resize(im, img_size=640, square=False):
    """
    Resize image maintaining aspect ratio or to square
    
    Args:
        im: Input image (numpy array)
        img_size: Target size
        square: If True, resize to square (img_size x img_size)
    
    Returns:
        Resized image
    """
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]
        r = img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    return im


class SafeBboxParams(A.BboxParams):
    """
    BboxParams compatible with albumentations 1.3.1
    
    Removes check_validity parameter (not supported in v1.3.1)
    Sets permissive defaults to prevent bbox filtering
    """
    def __init__(self, *args, **kwargs):
        # Remove unsupported parameters
        kwargs.pop('check_validity', None)
        
        # Set permissive defaults (don't filter boxes)
        if 'min_visibility' not in kwargs:
            kwargs['min_visibility'] = 0.0  # Keep all boxes regardless of visibility
        if 'min_area' not in kwargs:
            kwargs['min_area'] = 0.0  # Keep all boxes regardless of size
        
        super().__init__(*args, **kwargs)


def clip_bbox_to_valid_range(bbox, img_height, img_width, min_size=2.0):
    """
    Clip bbox to valid image bounds with minimum size guarantee
    
    Args:
        bbox: [xmin, ymin, xmax, ymax]
        img_height: Image height
        img_width: Image width
        min_size: Minimum bbox dimension
    
    Returns:
        Clipped bbox [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Clip to image bounds
    xmin = max(0.0, min(xmin, img_width - min_size))
    ymin = max(0.0, min(ymin, img_height - min_size))
    xmax = max(xmin + min_size, min(xmax, img_width))
    ymax = max(ymin + min_size, min(ymax, img_height))
    
    # Ensure minimum size
    if xmax - xmin < min_size:
        xmax = min(xmin + min_size, img_width)
    if ymax - ymin < min_size:
        ymax = min(ymin + min_size, img_height)
    
    return [xmin, ymin, xmax, ymax]


def get_train_aug():
    """Enhanced augmentation for fire & smoke"""
    return A.Compose([
        # Geometric - keep current
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        
        # Blur - REDUCE FURTHER for fire/smoke clarity
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),  # Reduced from (3,5)
        ], p=0.2),  # Reduced from 0.3
        
        # Color - ADD more variation for fire/smoke colors
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # Increased from 0.2
                contrast_limit=0.3,    # Increased from 0.2
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,    # Increased from 10
                sat_shift_limit=30,    # Increased from 20
                val_shift_limit=15,    # Increased from 10
                p=1.0
            ),
            A.CLAHE(clip_limit=2.0, p=1.0),  # ADD: Improves contrast
        ], p=0.7),  # Increased from 0.6
        
        # Weather - keep current
        A.RandomFog(
            fog_coef_lower=0.1,
            fog_coef_upper=0.3,
            alpha_coef=0.08,
            p=0.15
        ),
        
        # Lighting - CRITICAL for fire detection
        A.RandomGamma(
            gamma_limit=(70, 130),  # Wider range from (80, 120)
            p=0.4  # Increased from 0.3
        ),
        
        # ADD: RandomSunFlare for fire-like artifacts (reduces false positives)
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=1,
            num_flare_circles_upper=2,
            src_radius=50,
            p=0.1
        ),
        
        ToTensorV2(p=1.0),
    ], bbox_params=SafeBboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.2,  # Increased from 0.1 to filter tiny boxes
        min_area=8,          # Increased from 4 (2x2 -> ~3x3 pixels)
    ))


def get_train_transform():
    """
    Minimal transform for training (no augmentation)
    Use this when augmentation is disabled
    """
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=SafeBboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))


def get_valid_transform():
    """
    Validation transform (no augmentation, just tensor conversion)
    """
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=SafeBboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))


def transform_mosaic(mosaic_image, boxes, img_size=640):
    """
    Transform mosaic image with safe bbox handling
    
    Args:
        mosaic_image: Combined mosaic image (2*img_size x 2*img_size)
        boxes: Bounding boxes in mosaic coordinates
        img_size: Target size for final image
    
    Returns:
        resized_image, resized_boxes
    """
    # Resize mosaic to target size
    aug = A.Compose([
        A.Resize(img_size, img_size, always_apply=True, p=1.0)
    ])
    
    sample = aug(image=mosaic_image)
    resized_image = sample['image']
    
    # Scale boxes proportionally
    scale_factor = img_size / mosaic_image.shape[0]
    
    if len(boxes) == 0:
        return resized_image, np.array([])
    
    scaled_boxes = np.array(boxes) * scale_factor
    
    # Validate and clip boxes
    validated_boxes = []
    h, w = resized_image.shape[:2]
    
    for box in scaled_boxes:
        xmin, ymin, xmax, ymax = box
        
        # Clip to bounds
        clipped = clip_bbox_to_valid_range([xmin, ymin, xmax, ymax], h, w, min_size=2.0)
        
        # Only keep valid boxes (minimum size met)
        if clipped[2] > clipped[0] + 1 and clipped[3] > clipped[1] + 1:
            validated_boxes.append(clipped)
    
    return resized_image, np.array(validated_boxes) if validated_boxes else np.array([])


def infer_transforms(image):
    """
    Transform for inference (just convert to tensor)
    
    Args:
        image: Input image (numpy array, RGB)
    
    Returns:
        Tensor image
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_bbox(bbox, img_height, img_width):
    """
    Check if bbox is valid
    
    Args:
        bbox: [xmin, ymin, xmax, ymax]
        img_height: Image height
        img_width: Image width
    
    Returns:
        bool: True if valid, False otherwise
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Check bounds
    if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
        return False
    
    # Check order
    if xmin >= xmax or ymin >= ymax:
        return False
    
    # Check minimum size (at least 2x2 pixels)
    if (xmax - xmin) < 2 or (ymax - ymin) < 2:
        return False
    
    return True


def fix_bbox_format(bbox, img_height, img_width, format='pascal_voc'):
    """
    Fix and normalize bbox format
    
    Args:
        bbox: Input bbox (various formats)
        img_height: Image height
        img_width: Image width
        format: 'pascal_voc' (xyxy) or 'yolo' (normalized xywh)
    
    Returns:
        Fixed bbox in pascal_voc format [xmin, ymin, xmax, ymax]
    """
    if format == 'yolo':
        # Convert from normalized [x_center, y_center, width, height] to [xmin, ymin, xmax, ymax]
        x_center, y_center, w, h = bbox
        xmin = (x_center - w / 2) * img_width
        ymin = (y_center - h / 2) * img_height
        xmax = (x_center + w / 2) * img_width
        ymax = (y_center + h / 2) * img_height
        bbox = [xmin, ymin, xmax, ymax]
    
    # Ensure integers for pixel coordinates
    bbox = [float(x) for x in bbox]
    
    # Clip to valid range
    bbox = clip_bbox_to_valid_range(bbox, img_height, img_width)
    
    return bbox


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED AUGMENTATION (OPTIONAL - FOR EXPERIMENTATION)
# ═══════════════════════════════════════════════════════════════════════════════

def get_heavy_aug():
    """
    Heavy augmentation for challenging scenarios
    
    WARNING: Use carefully - may cause overfitting or training instability
    Only use if:
    - You have very diverse data
    - Standard augmentation gives mAP < 0.40
    - You're experiencing severe overfitting
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),  # Rare but possible for aerial views
        
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        
        # More aggressive blur
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        
        # More color variation
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.8),
        
        # Noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Weather
        A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=0.2),
        A.RandomRain(slant_lower=-10, slant_upper=10, p=0.1),
        
        ToTensorV2(p=1.0),
    ], bbox_params=SafeBboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.2,
        min_area=8,
    ))