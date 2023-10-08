import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result

from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

def detect(cfg,opt):

    logger = None
    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    # model.fuse()
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    # # flops and params
    # # ------------------------start--------------------------

    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 384, 640), as_strings=True,
    #                                             print_per_layer_stat=False, verbose=False)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # verbose = False
    # # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    # n_p = sum(x.numel() for x in model.parameters())  # number parameters
    # n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    # # if verbose:
    # #     print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
    # #     for i, (name, p) in enumerate(model.named_parameters()):
    # #         name = name.replace('module_list.', '')
    # #         print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
    # #             (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    # name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    # print(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients")
    # exit(0)
    # # ------------------------end--------------------------

    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        det_out, da_seg_out,ll_seg_out = model(img)
        inf_out, _ = det_out

        # Apply NMS
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)

        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, size=(h ,w ), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze()
        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict,  size=(h ,w ), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze()

        da_seg_mask = da_seg_mask-ll_seg_mask
        road_1 = torch.zeros_like(da_seg_mask)
        # road
        road_1[da_seg_mask == 1] = 1
        da_seg_mask = road_1
        da_seg_mask = da_seg_mask.cpu().numpy()
        ll_seg_mask = ll_seg_mask.cpu().numpy() 

        if dataset.mode == 'images':
            # # convert to BGR
            img_det = img_det[..., ::-1]
            
            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
            
            if len(det):
                det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{conf:.2f}'
                    plot_one_box(xyxy, img_det , label=label_det_pred, color=(0,255,255), line_thickness=2)       
            
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            img_det = img_det[..., ::-1]
            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
            if len(det):
                det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{conf:.2f}'
                    plot_one_box(xyxy, img_det , label=label_det_pred, color=(0,255,255), line_thickness=2)       
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

 
        # model inferring and postprocessing
        # for _ in range(20):
        #     with torch.no_grad():
        #         print('test1: model inferring and postprocessing')
        #         print('inferring 1 image for 10 times...')
        #         torch.cuda.synchronize()
        #         t1 = time.time()
        #         for _ in range(100):
        #             det_out, da_seg_out,ll_seg_out = model(img)
        #             inf_out, _ = det_out
        #             # det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        #         torch.cuda.synchronize()
        #         t2 = time.time()
        #         tact_time = (t2 - t1) / 100
        #         print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')


    print('Results saved to %s' % Path(opt.save_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/project/zhanjiao/YOLOP/yolopx/weights/epoch-195.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/project/zhanjiao/YOLOP/yolopx/inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='/project/zhanjiao/YOLOP/yolopx/inference/cs_output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
