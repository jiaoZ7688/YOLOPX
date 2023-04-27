<div align="center">
<h1> YOLOPX: Anchor-free Multi-Task learning Network for Panoptic Driving Perception </h1>

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/jiaoZ7688/YOLOPX/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>

  Jiao Zhan, Yarong Luo, Chi Guo, Yejun Wu, Jingnan Liu
</div>

## Paper

* The paper is under submission...

## News
* `2023-4-27`:  We've uploaded the experiment results along with some code, and the full code will be released soon!

## Introduction

 Panoptic driving perception involves a combination of traffic object detection, drivable area segmentation, and lane detection. Existing methods generally leverage multi-task learning networks built upon anchor-based detectors to deal with it, so will be sen-sitive to anchor design. We propose YOLOPX, to the best of our knowledge, the first anchor-free multi-task learning network ca-pable of addressing panoptic driving perception in real-time. Compared to prior works, we make several vital improvements. First, we replace YOLO’s detection head with a decoupled one to eliminate anchors. This anchor-free manner improves the network performance, simplifies the training process, and enhances the network's extensibility. Then, we combine multi-scale high-resolution features and self-attention for better lane detection. In addition to the above architecture optimization, we also propose a novel training strategy to improve training efficiency without anchors and additional inference cost. Our method achieves state-of-the-art (SOTA) performance on the challenging BDD100K dataset: It achieves 93.7 % recall and 83.3% mAP50 on traffic object detection, 93.2% mIoU on drivable area segmentation, and 88.6% accuracy and 27.2% IoU on lane detection. Moreover, YOLOPX has faster inference speed compared to the lightweight network YOLOP. Thus, YOLOPX is a powerful solu-tion for panoptic driving perception problems. The code is available at https://github.com/jiaoZ7688/YOLOPX.

## Results
We used the BDD100K as our datasets,and experiments are run on **NVIDIA TESLA V100**.
### ----------------------------
### video visualization Results
model : trained on the BDD100k train set and test on the BDD100k val set .

Note: The raw video comes from [YOLOP](https://github.com/hustvl/YOLOP/tree/main/inference/videos)
The results of our experiments are as follows:
<td><img src=demo/3.gif/></td>

Note: The raw video comes from [HybridNets](https://github.com/datvuthanh/HybridNets/tree/main/demo/video/)
The results of our experiments are as follows:
<td><img src=demo/2.gif/></td>

### ----------------------------
### image visualization Results
model : trained on the BDD100k train set and test on the BDD100k val set .
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


## Dataset
For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)


<!-- ## Demo Test

You can use the image or video as input.

```shell
python demo.py  --source demo/example.jpg
``` -->

## License

YOLOPX is released under the [MIT Licence](LICENSE).

