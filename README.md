<div align="center">
<h1> YOLOPX: Anchor-free Multi-Task learning Network for Panoptic Driving Perception </h1>

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/jiaoZ7688/YOLOPX/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.12+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>

  Jiao Zhan, Yarong Luo, Chi Guo, Yejun Wu, Jingnan Liu
</div>

## Paper

* If you find our work useful, please cite this paper: **Zhan J, Luo Y, Guo C, et al. YOLOPX: Anchor-free multi-task learning network for panoptic driving perception[J]. Pattern Recognition, 2024, 148: 110152.** [paper](https://www.sciencedirect.com/science/article/abs/pii/S003132032300849X?fr=RR-2&ref=pdf_download&rr=8abb0351d9898616)

## News
* `2023-4-27`:  We've uploaded the experiment results along with some code, and the full code will be released soon!

* `2023-9-15`:  We have uploaded part of the code and the full code will be released soon!

* `2024-10-24`:  We have released the full code!

## Introduction

Panoptic driving perception encompasses traffic object detection, drivable area segmentation, and lane detection. Existing methods typically utilize anchor-based multi-task learning networks to complete this task. While these methods yield promising results, they suffer from the inherent limitations of anchor-based detectors. In this paper, we propose YOLOPX, a simple and efficient anchor-free multi-task learning network for panoptic driving perception. To the best of our knowledge, this is the first work to employ the anchor-free detection head in panoptic driving perception. This anchor-free manner simplifies training by avoiding anchor-related heuristic tuning, and enhances the adaptability and scalability of our multi-task learning network. In addition, YOLOPX incorporates a novel lane detection head that combines multi-scale high-resolution features and long-distance contextual dependencies to improve segmentation performance. Beyond structure optimization, we propose optimization improvements to enhance network training, enabling our multi-task learning network to achieve optimal performance through simple end-to-end training. Experimental results on the challenging BDD100K dataset demonstrate the state-of-the-art (SOTA) performance of YOLOPX: it achieves 93.7 % recall and 83.3% mAP50 on traffic object detection, 93.2% mIoU on drivable area segmentation, and 88.6% accuracy and 27.2% IoU on lane detection. Moreover, YOLOPX has faster inference speed compared to the lightweight network YOLOP. Consequently, YOLOPX is a powerful solution for panoptic driving perception problems. The code is available at https://github.com/jiaoZ7688/YOLOPX.

## Results
* We use the BDD100K as our datasets,and experiments are run on **NVIDIA TESLA V100**.
* model : trained on the BDD100k train set and test on the BDD100k val set .
  
### ----------------------------
### video visualization Results

* **Our network has excellent robustness and generalization!!!!!**
* **Even on new datasets ([KITTI](https://www.cvlibs.net/datasets/kitti/)) with different image sizes and application scenarios, our network performs well.** 
* **This is helpful for related research in SLAM.**

* Note: The raw videos comes from [KITTI](https://www.cvlibs.net/datasets/kitti/)
* The results of our experiments are as follows:
<p><img src=demo/2_0.gif/><img src=demo/2_1.gif/><img src=demo/2_2.gif/><img src=demo/2_3.gif/></p>
  
* Note: The raw videos comes from [YOLOP](https://github.com/hustvl/YOLOP/tree/main/inference/videos) and [HybridNets](https://github.com/datvuthanh/HybridNets/tree/main/demo/video/)
* The results of our experiments are as follows:
<td><img src=demo/3.gif/></td>
<td><img src=demo/2.gif/></td>
 
### ----------------------------
### image visualization Results
* The results on the BDD100k val set.
<div align = 'None'>
  <img src="demo/1.jpg" width="100%" />
</div>

### Qualitative Comparison
<div align = 'None'>
  <img src="demo/all.jpg" width="100%" />
</div>


### Model parameter and inference speed
We compare YOLOPX with the current open source YOLOP and HybridNets on the NVIDIA RTX 3080. 
In terms of real-time, we compare the inference speed (excluding data pre-processing and NMS operations) at batch size 1.  


|        Model         |   Backbone   |   Params   | Speed (fps) |   Anchor    |
|:--------------------:|:------------:|:----------:|:-----------:|:-----------:|
|       `YOLOP`        |  CSPDarknet  |  **7.9M**  |     39      |     √      |
|     `HybridNets`     | EfficientNet |    12.8M   |     17      |     √      |
|   **`YOLOPX`**    |    ELANNet   |    32.9M   |   **47**    |   **×**    |


### Traffic Object Detection Result
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|        Model       |  Recall (%)  |   mAP@0.5 (%)   |
|:------------------:|:------------:|:---------------:|
|     `MultiNet`     |     81.3     |       60.2      |
|      `DLT-Net`     |     89.4     |       68.4      |
|   `Faster R-CNN`   |     77.2     |       55.6      |
|      `YOLOv5s`     |     86.8     |       77.2      |
|       `YOLOP`      |     89.2     |       76.5      |
|    `HybridNets`    |     92.8     |       77.3      |
|     `YOLOPv2`      |     91.1     |       **83.4**      |
|   **`YOLOPX`**    |   **93.7**   |     83.3    |

</td><td>

<img src="demo/det.jpg" width="100%" />

</td></tr> </table>


### Drivable Area Segmentation
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|       Model      | Drivable mIoU (%) |
|:----------------:|:-----------------:|
|    `MultiNet`    |        71.6       |
|     `DLT-Net`    |        71.3       |
|     `PSPNet`     |        89.6       |
|      `YOLOP`     |        91.5       |
|   `HybridNets`   |        90.5       |
|    `YOLOPv2`     |      **93.2**     |
|  **`YOLOPX`**   |      **93.2**     |

</td><td>

<img src="demo/da.jpg" width="100%" />

</td></tr> </table>


### Lane Line Detection
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|      Model       | Accuracy (%) | Lane Line IoU (%) |
|:----------------:|:------------:|:-----------------:|
|      `Enet`      |     34.12    |       14.64       |
|      `SCNN`      |     35.79    |       15.84       |
|    `Enet-SAD`    |     36.56    |       16.02       |
|      `YOLOP`     |     70.5     |       26.2        |
|    `HybridNets`  |     85.4     |     **31.6**      |
|     `YOLOPv2`    |     87.3     |       27.2        |
|   **`YOLOPX`**  |   **88.6**   |       27.2        |

</td><td>

<img src="demo/ll.jpg" width="100%" />

</td></tr> </table>


## Project Structure

```python
├─inference
│ ├─image   # inference images
│ ├─image_output   # inference result
├─lib
│ ├─config/default   # configuration of training and validation
│ ├─core    
│ │ ├─activations.py   # activation function
│ │ ├─evaluate.py   # calculation of metric
│ │ ├─function.py   # training and validation of model
│ │ ├─general.py   #calculation of metric、nms、conversion of data-format、visualization
│ │ ├─loss.py   # loss function
│ │ ├─postprocess.py   # postprocess(refine da-seg and ll-seg, unrelated to paper)
│ ├─dataset
│ │ ├─AutoDriveDataset.py   # Superclass dataset，general function
│ │ ├─bdd.py   # Subclass dataset，specific function
│ │ ├─convect.py 
│ │ ├─DemoDataset.py   # demo dataset(image, video and stream)
│ ├─models
│ │ ├─YOLOP.py    # Setup and Configuration of model
│ │ ├─YOLOX_Head.py    # YOLOX's decoupled Head
│ │ ├─YOLOX_Loss.py    # YOLOX's detection Loss
│ │ ├─commom.py   # calculation module
│ ├─utils
│ │ ├─augmentations.py    # data augumentation
│ │ ├─autoanchor.py   # auto anchor(k-means)
│ │ ├─split_dataset.py  # (Campus scene, unrelated to paper)
│ │ ├─plot.py  # plot_box_and_mask
│ │ ├─utils.py  # logging、device_select、time_measure、optimizer_select、model_save&initialize 、Distributed training
│ ├─run
│ │ ├─dataset/training time  # Visualization, logging and model_save
├─tools
│ │ ├─demo.py    # demo(folder、camera)
│ │ ├─test.py    
│ │ ├─train.py    
├─weights    # Pretraining model
```

---

## Requirement

This codebase has been developed with python version 3.7, PyTorch 1.12+ and torchvision 0.13+
```setup
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
or
```setup
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
See `requirements.txt` for additional dependencies and version requirements.
```setup
pip install -r requirements.txt
```

## Pre-trained Model
You can get the pre-trained model from <a href="https://pan.baidu.com/s/1k7_M8vrgQCnlY-FlaA0g6Q">here</a>.
Extraction code：fvuc

## Dataset
For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_seg_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `./lib/config/default.py`.

## Training

```shell
python tools/train.py
```

## Evaluation

```shell
python tools/test.py --weights weights/epoch-195.pth
```

## Demo

You can store the image or video in `--source`, and then save the reasoning result to `--save-dir`

```shell
python tools/demo.py --weights weights/epoch-195.pth
                     --source inference/image
                     --save-dir inference/image_output
                     --conf-thres 0.3
                     --iou-thres 0.45
```

## License

YOLOPX is released under the [MIT Licence](LICENSE).

## Acknowledgements

Our work would not be complete without the wonderful work of the following authors:

* [YOLOP](https://github.com/hustvl/YOLOP)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [HybridNets](https://github.com/datvuthanh/HybridNets)

