import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout
import albumentations as A
from collections import OrderedDict

from torch.nn.utils.rnn import pad_sequence

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        # albumentation data arguments
        self.albumentations_transform = A.Compose([

            A.OneOf([
                A.MotionBlur(p=0.1),
                A.MedianBlur(p=0.1),
                A.Blur(p=0.1),
            ], p=0.2),

            A.GaussNoise(p=0.02),
            A.CLAHE(p=0.02),
            A.RandomBrightnessContrast(p=0.02),
            A.RandomGamma(p=0.02),
            A.ImageCompression(quality_lower=75, p=0.02),

            A.OneOf([
                A.RandomSnow(p=0.1),  # 加雪花
                A.RandomRain(p=0.1),  # 加雨滴
                A.RandomFog(p=0.1),  # 加雾
                A.RandomSunFlare(p=0.1),  # 加阳光
                A.RandomShadow(p=0.1),  # 加阴影
            ], p=0.2),

            A.OneOf([
                A.ToGray(p=0.1),
                A.ToSepia(p=0.1),
            ], p=0.2),

            ],
            
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            additional_targets={'mask0': 'mask'})

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.mosaic_border = [-192, -320]

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

        self.mosaic_rate = cfg.mosaic_rate
        self.mixup_rate = cfg.mixup_rate
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def load_mosaic(self, idx):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        w_mosaic, h_mosaic = 640, 384

        yc = int(random.uniform(-self.mosaic_border[0], 2 * h_mosaic + self.mosaic_border[0])) # 192,3x192
        xc = int(random.uniform(-self.mosaic_border[1], 2 * w_mosaic + self.mosaic_border[1])) # 320,3x320
        
        indices = range(len(self.db))
        indices = [idx] + random.choices(indices, k=3)  # 3 additional iWmage indices
                        
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            # img, labels, seg_label, (h0,w0), (h, w), path = self.load_image(index), h=384, w = 640
            img, labels, seg_label, lane_label, (h0, w0), (h,w), path  = self.load_image(index)
                        
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((h_mosaic * 2, w_mosaic * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

                seg4 = np.full((h_mosaic * 2, w_mosaic * 2), 0, dtype=np.uint8)  # base image with 4 tiles

                lane4 = np.full((h_mosaic * 2, w_mosaic * 2), 0, dtype=np.uint8)  # base image with 4 tiles
                # 大图中左上角、右下角的坐标
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # 小图中左上角、右下角的坐标
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_mosaic * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_mosaic * 2), min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            seg4[y1a:y2a, x1a:x2a] = seg_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                
            lane4[y1a:y2a, x1a:x2a] = lane_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b
            
            if len(labels):
                labels[:, 1] += padw
                labels[:, 2] += padh
                labels[:, 3] += padw
                labels[:, 4] += padh
            
                labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        
        new = labels4.copy()
        new[:, 1:] = np.clip(new[:, 1:], 0, 2*w_mosaic)
        new[:, 2:5:2] = np.clip(new[:, 2:5:2], 0, 2*h_mosaic)

        # filter candidates
        i = box_candidates(box1=labels4[:,1:5].T, box2=new[:,1:5].T)
        labels4 = labels4[i]
        labels4[:] = new[i] 

        return img4, labels4, seg4, lane4, (h0, w0), (h, w), path

    def mixup(self, im, labels, seg_label, lane_label, im2, labels2, seg_label2, lane_label2 ):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        seg_label |= seg_label2
        lane_label |= lane_label2
        return im, labels, seg_label, lane_label

    def load_image(self, idx):
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # seg_label = cv2.imread(data["mask"], 0)
        if self.cfg.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0)
        lane_label = cv2.imread(data["lane"], 0)
        
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
       
        det_label = data["label"]
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()

            labels[:, 1] = (det_label[:, 1] - det_label[:, 3] / 2) * w
            labels[:, 2] = (det_label[:, 2] - det_label[:, 4] / 2) * h 
            labels[:, 3] = (det_label[:, 1] + det_label[:, 3] / 2) * w
            labels[:, 4] = (det_label[:, 2] + det_label[:, 4] / 2) * h
        
        return img, labels, seg_label, lane_label, (h0, w0), (h,w), data['image']

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """ 
        if self.is_train:
            mosaic_this = False
            if random.random() < self.mosaic_rate:
                mosaic_this = True
                #  this doubles training time with inherent stuttering in tqdm, prob cpu or io bottleneck, does prefetch_generator work with ddp? (no improvement)
                #  updated, mosaic is inherently slow, maybe cache the images in RAM? maybe it was IO bottleneck of reading 4 images everytime? time it
                img, labels, seg_label, lane_label, (h0, w0), (h, w), path = self.load_mosaic(idx)

                # mixup is double mosaic, really slow
                if random.random() < self.mixup_rate:
                    img2, labels2, seg_label2, lane_label2, (_, _), (_, _), _ = self.load_mosaic(random.randint(0, len(self.db) - 1))
                    img, labels, seg_label, lane_label = self.mixup(img, labels, seg_label, lane_label, img2, labels2, seg_label2, lane_label2)
            else:
                img, labels, seg_label, lane_label, (h0, w0), (h,w), path  = self.load_image(idx)

            try:
                new = self.albumentations_transform(image=img, mask=seg_label, mask0=lane_label,
                                                    bboxes=labels[:, 1:] if len(labels) else labels,
                                                    class_labels=labels[:, 0] if len(labels) else labels)
                img = new['image']
                labels = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]) if len(labels) else labels
                seg_label = new['mask']
                lane_label = new['mask0']
            except ValueError:  # bbox have width or height == 0
                pass

            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR,
                border=self.mosaic_border if mosaic_this else (0, 0)
            )

            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)

            # random left-right flip
            if random.random() < 0.5:
                img = np.fliplr(img)

                if len(labels):
                    rows, cols, channels = img.shape
                    x1 = labels[:, 1].copy()
                    x2 = labels[:, 3].copy()
                    x_tmp = x1.copy()
                    labels[:, 1] = cols - x2
                    labels[:, 3] = cols - x_tmp
                
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)

            # random up-down flip
            if random.random() < 0.0:
                img = np.flipud(img)

                if len(labels):
                    rows, cols, channels = img.shape
                    y1 = labels[:, 2].copy()
                    y2 = labels[:, 4].copy()
                    y_tmp = y1.copy()
                    labels[:, 2] = rows - y2
                    labels[:, 4] = rows - y_tmp

                seg_label = np.flipud(seg_label)
                lane_label = np.flipud(lane_label)
        
        else:
            img, labels, seg_label, lane_label, (h0, w0), (h,w), path = self.load_image(idx)
        
        (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), 640, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        if len(labels):
            # update labels after letterbox
            labels[:, 1] = ratio[0] * labels[:, 1] + pad[0]
            labels[:, 2] = ratio[1] * labels[:, 2] + pad[1]
            labels[:, 3] = ratio[0] * labels[:, 3] + pad[0]
            labels[:, 4] = ratio[1] * labels[:, 4] + pad[1]     

            # convert xyxy to ( cx, cy, w, h )
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

        labels_out = torch.zeros((len(labels), 5))
        if len(labels):
            labels_out[:, :] = torch.from_numpy(labels)

        img = np.ascontiguousarray(img)

        if self.cfg.num_seg_class == 3:
            _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
            _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        else:
            _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
        _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)

        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)

        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)

        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]),0)
            
        lane_label = torch.stack((lane2[0], lane1[0]),0)

        target = [labels_out, seg_label, lane_label]
        img = self.transform(img)

        return img, target, path, shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            # l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)

        label_det = pad_sequence(label_det, batch_first = True, padding_value = 0)

        return torch.stack(img, 0), [label_det, torch.stack(label_seg, 0), torch.stack(label_lane, 0)], paths, shapes


