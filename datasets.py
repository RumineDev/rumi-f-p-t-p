"""
IMPROVED DATASET - ROBUST DATA LOADING
=======================================

Key Improvements:
✓ Universal class handling (fire, smoke, other, or any custom classes)
✓ Comprehensive bbox validation and cleaning
✓ Safe augmentation application (no crashes)
✓ Handles both Pascal VOC (XML) and YOLO (TXT) formats
✓ Class balancing support via image_labels
✓ Better error handling and logging
✓ No bbox rejection (clipping instead)
"""

import torch
import cv2
import numpy as np
import os
import glob as glob
import random
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from utils.transforms import (
    get_train_transform, 
    get_valid_transform,
    get_train_aug,
    transform_mosaic,
    clip_bbox_to_valid_range,
    validate_bbox
)
from tqdm.auto import tqdm


class CustomDataset(Dataset):
    """
    Improved Custom Dataset for object detection
    
    Features:
    - Supports Pascal VOC (XML) and YOLO (TXT) formats
    - Robust bbox validation and fixing
    - Universal class handling
    - Safe augmentation
    - Class balancing support
    """
    
    def __init__(
        self, 
        images_path, 
        labels_path, 
        img_size, 
        classes, 
        transforms=None, 
        use_train_aug=False,
        train=False, 
        mosaic=0.3,
        square_training=False,
        label_type='pascal_voc'
    ):
        """
        Initialize dataset
        
        Args:
            images_path: Path to images directory
            labels_path: Path to labels directory
            img_size: Target image size
            classes: List of class names (first should be '__background__')
            transforms: Albumentations transforms
            use_train_aug: Whether to use training augmentation
            train: Training mode (enables mosaic)
            mosaic: Probability of mosaic augmentation (0.0-1.0)
            square_training: Resize to square (img_size x img_size)
            label_type: 'pascal_voc' or 'yolo'
        """
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.square_training = square_training
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG', '*.JPEG', '*.PNG']
        self.all_image_paths = []
        self.mosaic = mosaic
        self.label_type = label_type
        
        # Get all image paths
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        
        self.all_images = [os.path.basename(path) for path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        
        # Collect labels per image for WeightedRandomSampler (class balancing)
        print(f"Collecting image labels for class balancing...")
        self.image_labels = self._collect_image_labels()
        
        # Clean dataset (remove invalid images/annotations)
        if self.label_type == 'pascal_voc':
            self.read_and_clean()
        
        print(f"Dataset initialized: {len(self.all_images)} images")
    
    def _collect_image_labels(self):
        """
        Collect dominant class label for each image
        Used for class balancing via WeightedRandomSampler
        
        Returns:
            List of labels (one per image, -1 if no annotations)
        """
        image_labels = []
        
        for img_name in tqdm(self.all_images, desc="Collecting labels"):
            img_stem = os.path.splitext(img_name)[0]
            
            if self.label_type == 'pascal_voc':
                label_path = os.path.join(self.labels_path, f"{img_stem}.xml")
            else:
                label_path = os.path.join(self.labels_path, f"{img_stem}.txt")
            
            if not os.path.exists(label_path):
                image_labels.append(-1)
                continue
            
            try:
                if self.label_type == 'pascal_voc':
                    tree = ET.parse(label_path)
                    root = tree.getroot()
                    labels = []
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        if class_name in self.classes:
                            labels.append(self.classes.index(class_name))
                    
                    if labels:
                        # Use most common class in image
                        image_labels.append(max(set(labels), key=labels.count))
                    else:
                        image_labels.append(-1)
                else:
                    # YOLO format
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    if lines:
                        class_ids = [int(line.split()[0]) for line in lines if line.strip()]
                        if class_ids:
                            # Use most common class
                            image_labels.append(max(set(class_ids), key=class_ids.count))
                        else:
                            image_labels.append(-1)
                    else:
                        image_labels.append(-1)
            except Exception as e:
                print(f"Error reading label {label_path}: {e}")
                image_labels.append(-1)
        
        return image_labels
    
    def read_and_clean(self):
        """
        Validate and clean dataset (Pascal VOC only)
        
        Removes:
        - Images without annotations
        - Images with invalid bboxes (negative, zero-width, out-of-bounds)
        - Images that can't be loaded
        """
        print('Validating dataset...')
        images_to_remove = []
        issues_found = []
        
        for image_name in tqdm(self.all_images, desc="Validating"):
            img_stem = os.path.splitext(image_name)[0]
            label_path = os.path.join(self.labels_path, f"{img_stem}.xml")
            image_path = os.path.join(self.images_path, image_name)
            
            # Check if annotation exists
            if not os.path.exists(label_path):
                issues_found.append(f"Missing annotation: {image_name}")
                images_to_remove.append(image_name)
                continue
            
            # Check if image can be loaded
            try:
                img = cv2.imread(image_path)
                if img is None:
                    issues_found.append(f"Cannot load image: {image_name}")
                    images_to_remove.append(image_name)
                    continue
                img_h, img_w = img.shape[:2]
            except Exception as e:
                issues_found.append(f"Error loading {image_name}: {e}")
                images_to_remove.append(image_name)
                continue
            
            # Validate bboxes
            try:
                tree = ET.parse(label_path)
                root = tree.getroot()
                
                has_valid_box = False
                
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    
                    try:
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                    except (ValueError, AttributeError) as e:
                        issues_found.append(f"Invalid bbox format in {image_name}: {e}")
                        continue
                    
                    # Check bbox validity
                    if validate_bbox([xmin, ymin, xmax, ymax], img_h, img_w):
                        has_valid_box = True
                    else:
                        issues_found.append(f"Invalid bbox in {image_name}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                
                if not has_valid_box:
                    images_to_remove.append(image_name)
                    
            except Exception as e:
                issues_found.append(f"Error parsing {label_path}: {e}")
                images_to_remove.append(image_name)
        
        # Remove problematic images
        if images_to_remove:
            print(f"\n⚠️  Found {len(images_to_remove)} problematic images")
            if len(issues_found) <= 20:
                for issue in issues_found:
                    print(f"  • {issue}")
            else:
                for issue in issues_found[:20]:
                    print(f"  • {issue}")
                print(f"  ... and {len(issues_found) - 20} more issues")
            
            self.all_images = [img for img in self.all_images if img not in images_to_remove]
            print(f"✓ Removed {len(images_to_remove)} images")
            print(f"✓ Remaining: {len(self.all_images)} images")
        else:
            print(f"✓ All {len(self.all_images)} images valid")
    
    def resize(self, im, square=False):
        """Resize image maintaining aspect ratio or to square"""
        if square:
            im = cv2.resize(im, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        else:
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return im
    
    def load_image_and_labels(self, index):
        """
        Load image and labels with robust error handling
        
        Returns:
            orig_image, resized_image, orig_boxes, boxes, labels, area, iscrowd, dims
        """
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self.resize(image, square=self.square_training)
        image_resized /= 255.0
        
        # Load labels based on format
        if self.label_type == 'pascal_voc':
            return self.load_pascal_voc(image, image_name, image_resized)
        elif self.label_type == 'yolo':
            return self.load_yolo(image, image_name, image_resized)
        else:
            raise ValueError(f"Unsupported label_type: {self.label_type}")
    
    def load_pascal_voc(self, image, image_name, image_resized):
        """Load Pascal VOC format labels"""
        img_stem = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.labels_path, f"{img_stem}.xml")
        
        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        tree = ET.parse(label_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Universal class handling - keep ALL classes
            if class_name not in self.classes:
                continue
            
            labels.append(self.classes.index(class_name))
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Clip to valid range
            orig_box = clip_bbox_to_valid_range(
                [xmin, ymin, xmax, ymax],
                image_height,
                image_width
            )
            orig_boxes.append(orig_box)
            
            # Scale to resized image
            scale_x = image_resized.shape[1] / image_width
            scale_y = image_resized.shape[0] / image_height
            
            resized_box = [
                orig_box[0] * scale_x,
                orig_box[1] * scale_y,
                orig_box[2] * scale_x,
                orig_box[3] * scale_y
            ]
            
            # Clip resized box
            resized_box = clip_bbox_to_valid_range(
                resized_box,
                image_resized.shape[0],
                image_resized.shape[1]
            )
            boxes.append(resized_box)
        
        # Convert to tensors
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes_length > 0 else torch.zeros((0, 4), dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.zeros(0, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.zeros(0, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        
        return image, image_resized, orig_boxes, boxes, labels, area, iscrowd, (image_width, image_height)
    
    def load_yolo(self, image, image_name, image_resized):
        """Load YOLO format labels"""
        img_stem = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.labels_path, f"{img_stem}.txt")
        
        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        if not os.path.exists(label_path):
            # No annotations - return empty
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)
            return image, image_resized, [], boxes, labels, area, iscrowd, (image_width, image_height)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, norm_xc, norm_yc, norm_w, norm_h = parts
            class_id = int(class_id)
            norm_xc, norm_yc, norm_w, norm_h = map(float, [norm_xc, norm_yc, norm_w, norm_h])
            
            # Universal class handling
            if class_id >= len(self.classes) - 1:  # -1 for background class
                continue
            
            labels.append(class_id + 1)  # +1 because class 0 is background
            
            # Convert from YOLO format to Pascal VOC
            xc = norm_xc * image_width
            yc = norm_yc * image_height
            w = norm_w * image_width
            h = norm_h * image_height
            
            xmin = xc - w / 2
            ymin = yc - h / 2
            xmax = xc + w / 2
            ymax = yc + h / 2
            
            # Clip to valid range
            orig_box = clip_bbox_to_valid_range(
                [xmin, ymin, xmax, ymax],
                image_height,
                image_width
            )
            orig_boxes.append(orig_box)
            
            # Scale to resized image
            scale_x = image_resized.shape[1] / image_width
            scale_y = image_resized.shape[0] / image_height
            
            resized_box = [
                orig_box[0] * scale_x,
                orig_box[1] * scale_y,
                orig_box[2] * scale_x,
                orig_box[3] * scale_y
            ]
            
            # Clip resized box
            resized_box = clip_bbox_to_valid_range(
                resized_box,
                image_resized.shape[0],
                image_resized.shape[1]
            )
            boxes.append(resized_box)
        
        # Convert to tensors
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes_length > 0 else torch.zeros((0, 4), dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.zeros(0, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.zeros(0, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        
        return image, image_resized, orig_boxes, boxes, labels, area, iscrowd, (image_width, image_height)
    
    def load_cutmix_image_and_boxes(self, index):
        """
        Load mosaic (4-image) augmentation with safe bbox handling
        """
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]
        
        result_image = np.full((s * 2, s * 2, 3), 114/255, dtype=np.float32)
        result_boxes = []
        result_classes = []
        
        for i, idx in enumerate(indices):
            _, img_res, _, boxes, labels, _, _, _ = self.load_image_and_labels(idx)
            h, w = img_res.shape[:2]
            
            # Calculate placement
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # Place image
            result_image[y1a:y2a, x1a:x2a] = img_res[y1b:y2b, x1b:x2b]
            
            # Adjust boxes
            padw = x1a - x1b
            padh = y1a - y1b
            
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                result_boxes.append(boxes)
                result_classes.extend(labels.tolist())
        
        # Combine and validate boxes
        if result_boxes:
            result_boxes = torch.cat(result_boxes, dim=0)
            
            # Clip to mosaic bounds
            result_boxes[:, [0, 2]] = torch.clamp(result_boxes[:, [0, 2]], 0, 2 * s)
            result_boxes[:, [1, 3]] = torch.clamp(result_boxes[:, [1, 3]], 0, 2 * s)
            
            # Filter valid boxes
            valid = (result_boxes[:, 2] > result_boxes[:, 0] + 1) & (result_boxes[:, 3] > result_boxes[:, 1] + 1)
            result_boxes = result_boxes[valid]
            result_classes = [result_classes[i] for i in range(len(result_classes)) if valid[i]]
        
        # Resize mosaic to target size
        result_image, resized_boxes = transform_mosaic(result_image, result_boxes.numpy() if len(result_boxes) > 0 else [], self.img_size)
        
        return result_image, \
               torch.as_tensor(resized_boxes, dtype=torch.float32), \
               torch.as_tensor(result_classes, dtype=torch.int64), \
               torch.zeros(0), \
               torch.zeros(0), \
               (0, 0)
    
    def __getitem__(self, idx):
        """
        Get item with robust augmentation handling
        """
        # Load data
        if not self.train or random.random() >= self.mosaic:
            # No mosaic
            _, image_resized, _, boxes, labels, area, iscrowd, dims = self.load_image_and_labels(idx)
        else:
            # With mosaic
            image_resized, boxes, labels, area, iscrowd, dims = self.load_cutmix_image_and_boxes(idx)
        
        # Prepare target
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx])
        }
        
        # Handle empty boxes
        if boxes.numel() == 0:
            empty_aug = get_valid_transform()
            sample = empty_aug(image=image_resized, bboxes=[], labels=[])
            image_resized = sample['image']
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            return image_resized, target
        
        # Convert to lists for albumentations
        h, w = image_resized.shape[:2]
        bboxes_list = boxes.cpu().numpy().tolist()
        labels_list = labels.cpu().numpy().tolist()
        
        # Clip boxes to image bounds
        clipped_bboxes = []
        clipped_labels = []
        
        for box, label in zip(bboxes_list, labels_list):
            clipped = clip_bbox_to_valid_range(box, h, w, min_size=2.0)
            
            # Only keep valid boxes
            if clipped[2] > clipped[0] + 1 and clipped[3] > clipped[1] + 1:
                clipped_bboxes.append(clipped)
                clipped_labels.append(label)
        
        # Fallback if all boxes were filtered
        if not clipped_bboxes:
            empty_aug = get_valid_transform()
            sample = empty_aug(image=image_resized, bboxes=[], labels=[])
            image_resized = sample['image']
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            return image_resized, target
        
        # Apply augmentation
        if self.use_train_aug:
            aug = get_train_aug()
        else:
            aug = self.transforms
        
        try:
            sample = aug(image=image_resized, bboxes=clipped_bboxes, labels=clipped_labels)
        except Exception as e:
            # Fallback to no augmentation on error
            print(f"Warning: Augmentation failed for image {idx}: {e}")
            fallback = get_valid_transform()
            sample = fallback(image=image_resized, bboxes=clipped_bboxes, labels=clipped_labels)
        
        # Update target with augmented data
        image_resized = sample['image']
        target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32) if sample['bboxes'] else torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64) if sample['labels'] else torch.zeros(0, dtype=torch.int64)
        
        return image_resized, target
    
    def __len__(self):
        return len(self.all_images)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """Custom collate function for batching"""
    return tuple(zip(*batch))


def create_train_dataset(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    use_train_aug=False,
    mosaic=0.3,
    square_training=False,
    label_type='pascal_voc'
):
    """Create training dataset"""
    return CustomDataset(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True, 
        mosaic=mosaic,
        square_training=square_training,
        label_type=label_type
    )


def create_valid_dataset(
    valid_dir_images, 
    valid_dir_labels, 
    img_size, 
    classes,
    square_training=False,
    label_type='pascal_voc'
):
    """Create validation dataset"""
    return CustomDataset(
        valid_dir_images, 
        valid_dir_labels, 
        img_size, 
        classes, 
        get_valid_transform(),
        train=False, 
        square_training=square_training,
        label_type=label_type
    )


def create_train_loader(train_dataset, batch_size, num_workers=0, batch_sampler=None):
    """Create training dataloader"""
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )


def create_valid_loader(valid_dataset, batch_size, num_workers=0, batch_sampler=None):
    """Create validation dataloader"""
    return DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )