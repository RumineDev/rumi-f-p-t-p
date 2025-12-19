import albumentations as A
import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms

def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

# ERROR-PROOF Training Augmentation
def get_train_aug():
    """
    Safe augmentation pipeline that prevents bbox errors
    Key changes:
    - Higher min_visibility (0.3 instead of 0.2)
    - Larger min_area (25 instead of 16) 
    - Reduced RandomResizedCrop scale (0.90 min instead of 0.85)
    - Lower probabilities for aggressive transforms
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        
        # SAFE: Reduced shift/scale/rotate limits
        A.ShiftScaleRotate(
            shift_limit=0.03,      # Keep small shifts
            scale_limit=0.08,      # Reduced from 0.10
            rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3                  # Reduced probability
        ),
        
        # SAFE: Less aggressive crop
        A.RandomResizedCrop(
            height=640, width=640,
            scale=(0.90, 1.0),     # CRITICAL: 0.90 min (was 0.85)
            ratio=(0.95, 1.05),    # CRITICAL: Tighter ratio (was 0.9-1.1)
            p=0.2                  # Reduced probability
        ),
        
        # Safe blur augmentations
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.4),
        
        # Safe color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.25,
                p=0.6
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.015,
                p=0.4
            ),
        ], p=0.5),
        
        # Safe environmental effects
        A.RandomFog(
            alpha_coef=0.04,
            fog_coef_lower=0.1,
            fog_coef_upper=0.3,
            p=0.15                 # Reduced
        ),
        
        A.RandomGamma(
            gamma_limit=(85, 120),
            p=0.2                  # Reduced
        ),
        
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,        # CRITICAL: Increased (was 0.2)
        min_area=25                # CRITICAL: Increased (was 16) - 5x5 pixels
    ))


# Minimal augmentation (for testing/debugging)
def get_train_aug_minimal():
    """
    Ultra-safe augmentation for debugging
    Only horizontal flip - no bbox modifications
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1,
        min_area=4
    ))


def get_train_transform():
    """No augmentation - just convert to tensor"""
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))


def transform_mosaic(mosaic, boxes, img_size=640):
    """
    SAFE mosaic transformation with bbox validation
    """
    aug = A.Compose(
        [A.Resize(img_size, img_size, always_apply=True, p=1.0)]
    )
    sample = aug(image=mosaic)
    resized_mosaic = sample['image']
    
    # Transform boxes
    transformed_boxes = (np.array(boxes) / mosaic.shape[0]) * resized_mosaic.shape[1]
    
    # SAFE bbox correction with margin
    MARGIN = 2
    validated_boxes = []
    
    for box in transformed_boxes:
        xmin, ymin, xmax, ymax = box
        
        # Ensure minimum size
        if xmax - xmin < 4:
            if xmin + 4 <= resized_mosaic.shape[1] - MARGIN:
                xmax = xmin + 4
            else:
                continue  # Skip box if can't fit
        
        if ymax - ymin < 4:
            if ymin + 4 <= resized_mosaic.shape[0] - MARGIN:
                ymax = ymin + 4
            else:
                continue
        
        # Clip to image boundaries with margin
        xmin = max(MARGIN, min(xmin, resized_mosaic.shape[1] - MARGIN))
        ymin = max(MARGIN, min(ymin, resized_mosaic.shape[0] - MARGIN))
        xmax = max(xmin + 4, min(xmax, resized_mosaic.shape[1] - MARGIN))
        ymax = max(ymin + 4, min(ymax, resized_mosaic.shape[0] - MARGIN))
        
        # Final validation
        if xmax > xmin + 2 and ymax > ymin + 2:
            validated_boxes.append([xmin, ymin, xmax, ymax])
    
    return resized_mosaic, np.array(validated_boxes) if validated_boxes else np.array([])


def get_valid_transform():
    """Validation transform - no augmentation"""
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))


def infer_transforms(image):
    """Inference transform"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)