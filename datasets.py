import torch
import cv2
import numpy as np
import os
import glob as glob
import random
from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils.transforms import (
    get_train_transform, 
    get_valid_transform,
    get_train_aug,
    transform_mosaic
)
from tqdm.auto import tqdm


# the dataset class
class CustomDataset(Dataset):
    def __init__(
        self, 
        images_path, 
        labels_path, 
        img_size, 
        classes, 
        transforms=None, 
        use_train_aug=False,
        train=False, 
        mosaic=1.0,
        square_training=False,
        label_type='pascal_voc'
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.square_training = square_training
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.mosaic = mosaic
        self.log_annot_issue_y = True
        self.label_type = label_type
        
        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

        # Collect labels per image for WeightedRandomSampler
        self.image_labels = []
        for img in self.all_images:
            xml_path = os.path.join(self.labels_path, os.path.splitext(img)[0] + ".xml")
            if not os.path.exists(xml_path):
                self.image_labels.append(-1)
                continue
            tree = et.parse(xml_path)
            root = tree.getroot()
            labels = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in self.classes:
                    labels.append(self.classes.index(class_name))
            if len(labels) > 0:
                self.image_labels.append(labels[0])
            else:
                self.image_labels.append(-1)

        # Remove images with no annotations or invalid boxes
        if self.label_type == 'pascal_voc':
            self.read_and_clean()

    def read_and_clean(self):
        print('Checking Labels and images...')
        images_to_remove = []
        problematic_images = []

        for image_name in tqdm(self.all_images, total=len(self.all_images)):
            possible_annot_name = os.path.join(self.labels_path, os.path.splitext(image_name)[0] + '.xml')
            if not os.path.exists(possible_annot_name):
                print(f"⚠️ Annotation not found: {possible_annot_name}. Removing {image_name}")
                images_to_remove.append(image_name)
                continue

            tree = et.parse(possible_annot_name)
            root = tree.getroot()
            invalid_bbox = False

            for member in root.findall('object'):
                try:
                    xmin = float(member.find('bndbox').find('xmin').text)
                    xmax = float(member.find('bndbox').find('xmax').text)
                    ymin = float(member.find('bndbox').find('ymin').text)
                    ymax = float(member.find('bndbox').find('ymax').text)
                except (ValueError, AttributeError):
                    invalid_bbox = True
                    break

                if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                    invalid_bbox = True
                    break

            if invalid_bbox:
                problematic_images.append(image_name)
                images_to_remove.append(image_name)

        # Filter out problematic
        self.all_images = [img for img in self.all_images if img not in images_to_remove]
        self.all_annot_paths = [
            path for path in self.all_annot_paths 
            if os.path.splitext(os.path.basename(path))[0] + '.xml' not in [
                os.path.splitext(img)[0] + '.xml' for img in images_to_remove
            ]
        ]

        if problematic_images:
            print("\n⚠️ The following images have invalid bounding boxes and will be removed:")
            for img in problematic_images:
                print(f"⚠️ {img}")

        print(f"Removed {len(images_to_remove)} problematic images and annotations.")

    def resize(self, im, square=False):
        if square:
            im = cv2.resize(im, (self.img_size, self.img_size))
        else:
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
        return im

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self.resize(image, square=self.square_training)
        image_resized /= 255.0

        if self.label_type == 'pascal_voc':
            return self.load_pascal_voc(image, image_name, image_resized)
        elif self.label_type == 'yolo':
            return self.load_yolo(image, image_name, image_resized)
        else:
            raise ValueError("Unsupported label_type")

    def load_pascal_voc(self, image, image_name, image_resized):
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]

        tree = et.parse(annot_file_path)
        root = tree.getroot()
        for member in root.findall('object'):
            class_name = member.find('name').text
            if class_name not in self.classes:
                continue
            labels.append(self.classes.index(class_name))

            xmin = float(member.find('bndbox').find('xmin').text)
            xmax = float(member.find('bndbox').find('xmax').text)
            ymin = float(member.find('bndbox').find('ymin').text)
            ymax = float(member.find('bndbox').find('ymax').text)

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, ymin, xmax, ymax,
                image_width, image_height, orig_data=True
            )
            orig_boxes.append([xmin, ymin, xmax, ymax])

            # Scale to resized image
            xmin = (xmin / image_width) * image_resized.shape[1]
            xmax = (xmax / image_width) * image_resized.shape[1]
            ymin = (ymin / image_height) * image_resized.shape[0]
            ymax = (ymax / image_height) * image_resized.shape[0]

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, ymin, xmax, ymax,
                image_resized.shape[1], image_resized.shape[0], orig_data=False
            )
            boxes.append([xmin, ymin, xmax, ymax])

        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes_length > 0 else torch.zeros((0, 4), dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.zeros(0, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.zeros(0, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        return image, image_resized, orig_boxes, boxes, labels, area, iscrowd, (image_width, image_height)

    def load_yolo(self, image, image_name, image_resized):
        annot_filename = os.path.splitext(image_name)[0] + '.txt'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        image_width = image.shape[1]
        image_height = image.shape[0]

        if not os.path.exists(annot_file_path):
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)
            return image, image_resized, [], boxes, labels, area, iscrowd, (image_width, image_height)

        with open(annot_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            label, norm_xc, norm_yc, norm_w, norm_h = parts
            label = int(label)
            norm_xc, norm_yc, norm_w, norm_h = map(float, [norm_xc, norm_yc, norm_w, norm_h])

            if label >= len(self.classes):
                continue

            labels.append(label)
            xc = norm_xc * image_width
            yc = norm_yc * image_height
            w = norm_w * image_width
            h = norm_h * image_height

            xmin = xc - w / 2
            ymin = yc - h / 2
            xmax = xc + w / 2
            ymax = yc + h / 2

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, ymin, xmax, ymax,
                image_width, image_height, orig_data=True
            )
            orig_boxes.append([xmin, ymin, xmax, ymax])

            xmin = (xmin / image_width) * image_resized.shape[1]
            xmax = (xmax / image_width) * image_resized.shape[1]
            ymin = (ymin / image_height) * image_resized.shape[0]
            ymax = (ymax / image_height) * image_resized.shape[0]

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, ymin, xmax, ymax,
                image_resized.shape[1], image_resized.shape[0], orig_data=False
            )
            boxes.append([xmin, ymin, xmax, ymax])

        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes_length > 0 else torch.zeros((0, 4), dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.zeros(0, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.zeros(0, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        return image, image_resized, orig_boxes, boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(self, xmin, ymin, xmax, ymax, width, height, orig_data=False):
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)

        if xmax - xmin <= 1.0:
            xmax = xmin + 1.0
        if ymax - ymin <= 1.0:
            ymax = ymin + 1.0

        return xmin, ymin, xmax, ymax

    def load_cutmix_image_and_boxes(self, index):
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        result_image = np.full((s * 2, s * 2, 3), 114/255, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, idx in enumerate(indices):
            _, img_res, _, boxes, labels, _, _, _ = self.load_image_and_labels(idx)
            h, w = img_res.shape[:2]

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = img_res[y1b:y2b, x1b:x2b]

            padw = x1a - x1b
            padh = y1a - y1b

            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                result_boxes.append(boxes)
                result_classes.extend(labels.tolist())

        if result_boxes:
            result_boxes = torch.cat(result_boxes, dim=0)
            result_boxes[:, [0, 2]] = torch.clamp(result_boxes[:, [0, 2]], 0, 2 * s)
            result_boxes[:, [1, 3]] = torch.clamp(result_boxes[:, [1, 3]], 0, 2 * s)

            # Filter valid boxes
            valid = (result_boxes[:, 2] > result_boxes[:, 0]) & (result_boxes[:, 3] > result_boxes[:, 1])
            result_boxes = result_boxes[valid]
            result_classes = [result_classes[i] for i in range(len(result_classes)) if valid[i]]

        # Resize mosaic to final size
        result_image, resized_boxes = transform_mosaic(result_image, result_boxes.numpy(), self.img_size)
        return result_image, torch.as_tensor(resized_boxes, dtype=torch.float32), \
               torch.as_tensor(result_classes, dtype=torch.int64), \
               torch.zeros(0), torch.zeros(0), (0, 0)

    def __getitem__(self, idx):
        if not self.train:
            _, image_resized, _, boxes, labels, area, iscrowd, dims = self.load_image_and_labels(idx)
        else:
            if random.random() < self.mosaic:
                image_resized, boxes, labels, area, iscrowd, dims = self.load_cutmix_image_and_boxes(idx)
            else:
                _, image_resized, _, boxes, labels, area, iscrowd, dims = self.load_image_and_labels(idx)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx])
        }

        # Handle empty boxes
        if boxes.numel() == 0:
            # Use dummy augmentation (just ToTensor)
            empty_aug = get_valid_transform()
            sample = empty_aug(image=image_resized, bboxes=[], labels=[])
            image_resized = sample['image']
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            return image_resized, target

        # ✅ Convert to plain Python lists before Albumentations
        bboxes_list = boxes.cpu().numpy().tolist()
        labels_list = labels.cpu().numpy().tolist()

        # ✅ Clip to image bounds (pixel coordinates)
        h, w = image_resized.shape[:2]
        clipped_bboxes = []
        clipped_labels = []
        for box, label in zip(bboxes_list, labels_list):
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(x1 + 1.0, min(x2, w))
            y2 = max(y1 + 1.0, min(y2, h))
            if x2 > x1 and y2 > y1:
                clipped_bboxes.append([x1, y1, x2, y2])
                clipped_labels.append(label)

        if not clipped_bboxes:
            # Fallback to empty
            empty_aug = get_valid_transform()
            sample = empty_aug(image=image_resized, bboxes=[], labels=[])
            image_resized = sample['image']
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            return image_resized, target

        # ✅ Apply augmentation with safe inputs
        if self.use_train_aug:
            aug = get_train_aug()
        else:
            aug = self.transforms

        try:
            sample = aug(image=image_resized, bboxes=clipped_bboxes, labels=clipped_labels)
        except Exception as e:
            # Fallback to no-aug if aug fails
            fallback = get_valid_transform()
            sample = fallback(image=image_resized, bboxes=[], labels=[])
            image_resized = sample['image']
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            return image_resized, target

        image_resized = sample['image']
        target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def collate_fn(batch):
    return tuple(zip(*batch))


def create_train_dataset(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    use_train_aug=False,
    mosaic=1.0,
    square_training=False,
    label_type='pascal_voc'
):
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
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )


def create_valid_loader(valid_dataset, batch_size, num_workers=0, batch_sampler=None):
    return DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )