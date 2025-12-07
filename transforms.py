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

# Define the training tranforms
# Enhanced augmentation for fire/smoke detection with better handling of class imbalance
def get_train_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),  # Added for more diversity
        A.ShiftScaleRotate(
            shift_limit=0.05,  # Increased from 0.03
            scale_limit=0.15,  # Increased from 0.10 for better scale variation
            rotate_limit=10,   # Increased from 5 for better rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5  # Increased from 0.35
        ),
        A.RandomResizedCrop(
            height=640, width=640,
            scale=(0.8, 1.0),  # Increased range from (0.85, 1.0)
            ratio=(0.85, 1.15),  # Increased range from (0.9, 1.1)
            p=0.35  # Increased from 0.25
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.4),  # Increased from 3
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),  # Increased from (3, 5)
            A.MedianBlur(blur_limit=5, p=0.2),  # Increased from 3
        ], p=0.5),  # Increased from 0.45
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.25,  # Increased from 0.20
                contrast_limit=0.30,   # Increased from 0.25
                p=0.7  # Increased from 0.6
            ),
            A.ColorJitter(
                brightness=0.20,  # Increased from 0.15
                contrast=0.20,    # Increased from 0.15
                saturation=0.15,  # Increased from 0.10
                hue=0.02,         # Increased from 0.015
                p=0.5  # Increased from 0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.3
            ),
        ], p=0.65),  # Increased from 0.5
        A.RandomFog(
            alpha_coef=0.05,  # Increased from 0.04
            fog_coef_lower=0.1,
            fog_coef_upper=0.4,  # Increased from 0.3 - helps with smoke detection
            p=0.30  # Increased from 0.20
        ),
        A.RandomGamma(
            gamma_limit=(80, 130),  # Increased range from (85, 120)
            p=0.30  # Increased from 0.25
        ),
        # CLAHE removed - requires uint8 input but image is normalized float
        # Using RandomBrightnessContrast and ColorJitter instead for contrast enhancement
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            p=0.2  # Added for more realistic variations
        ),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.15  # Reduced from 0.20 to keep more bboxes after augmentation
    ))


def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def transform_mosaic(mosaic, boxes, img_size=640):
    """
    Resizes the `mosaic` image to `img_size` which is the desired image size
    for the neural network input. Also transforms the `boxes` according to the
    `img_size`.

    :param mosaic: The mosaic image, Numpy array.
    :param boxes: Boxes Numpy.
    :param img_resize: Desired resize.
    """
    aug = A.Compose(
        [A.Resize(img_size, img_size, always_apply=True, p=1.0)
    ])
    sample = aug(image=mosaic)
    resized_mosaic = sample['image']
    transformed_boxes = (np.array(boxes) / mosaic.shape[0]) * resized_mosaic.shape[1]
    for box in transformed_boxes:
        # Bind all boxes to correct values. This should work correctly most of
        # of the time. There will be edge cases thought where this code will
        # mess things up. The best thing is to prepare the dataset as well as 
        # as possible.
        if box[2] - box[0] <= 1.0:
            box[2] = box[2] + (1.0 - (box[2] - box[0]))
            if box[2] >= float(resized_mosaic.shape[1]):
                box[2] = float(resized_mosaic.shape[1])
        if box[3] - box[1] <= 1.0:
            box[3] = box[3] + (1.0 - (box[3] - box[1]))
            if box[3] >= float(resized_mosaic.shape[0]):
                box[3] = float(resized_mosaic.shape[0])
    return resized_mosaic, transformed_boxes

# Define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)
