# YCDa: YCbCr Decoupled Attention for Real-time Realistic Camouflaged Object Detection

![YCDa](images/performence_overview.png)

## Overview

Human vision exhibits remarkable adaptability in perceiving objects under camouflage. When color cues become unreliable, the visual system instinctively shifts its reliance from chrominance (color) to luminance (brightness and texture), enabling more robust perception in visually confusing environments. Drawing inspiration from this biological mechanism, we propose **YCDa**, an efficient early-stage feature processing strategy that embeds this “chrominance–luminance decoupling and dynamic attention” principle into modern real-time detectors. Specifically, YCDa separates color and luminance information in the input stage and dynamically allocates attention across channels to amplify discriminative cues while suppressing misleading color noise. The strategy is plug-and-play and can be integrated into existing detectors by simply replacing the first downsampling layer. Extensive experiments on multiple baselines demonstrate that YCDa consistently improves performance with negligible overhead as shown in the figure above. Notably, **YCDa-YOLO12s** achieves a **112% improvement in mAP** over the baseline on COD10K-D and sets new state-of-the-art results for real-time camouflaged object detection across COD-D datasets.

![YCDa](images/YCDa.png)
![YCDa](images/Visualization.png)

## Performance Comparison

**Table: Performance comparison between YCDa-enhanced models and baselines on COD10K-D, NC4K-D, and CAMO-D datasets.** Bold indicates the best-performing method within each baseline group.
<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="3">COD10K-D</th>
      <th colspan="3">NC4K-D</th>
      <th colspan="3">CAMO-D</th>
    </tr>
    <tr>
      <th>mAP</th><th>AP50</th><th>AP75</th>
      <th>mAP</th><th>AP50</th><th>AP75</th>
      <th>mAP</th><th>AP50</th><th>AP75</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RT-DETR-L</td>
      <td>18.4</td><td>26.6</td><td>19.3</td>
      <td><b>35.9</b></td><td><b>50.8</b></td><td><b>36.9</b></td>
      <td><b>29.3</b></td><td><b>39.5</b></td><td><b>31.2</b></td>
    </tr>
    <tr>
      <td>RT-DETR-L + YCDa</td>
      <td><b>19.6</b></td><td><b>28.3</b></td><td><b>20.7</b></td>
      <td>34.1</td><td>48.7</td><td>34.0</td>
      <td>22.5</td><td>30.1</td><td>23.1</td>
    </tr>
    <tr>
      <td>YOLOv8s</td>
      <td>11.2</td><td>19.9</td><td>10.9</td>
      <td>29.0</td><td>45.1</td><td>29.9</td>
      <td>20.5</td><td>31.1</td><td>21.2</td>
    </tr>
    <tr>
      <td>YOLOv8s + YCDa</td>
      <td><b>14.7</b></td><td><b>24.4</b></td><td><b>15.1</b></td>
      <td><b>32.0</b></td><td><b>48.6</b></td><td><b>32.8</b></td>
      <td><b>21.6</b></td><td><b>31.7</b></td><td><b>21.5</b></td>
    </tr>
    <tr>
      <td>YOLO11s</td>
      <td>10.4</td><td>17.5</td><td>10.5</td>
      <td>27.0</td><td>40.5</td><td>28.3</td>
      <td>21.4</td><td>31.3</td><td>21.7</td>
    </tr>
    <tr>
      <td>YOLO11s + YCDa</td>
      <td><b>17.2</b></td><td><b>26.3</b></td><td><b>18.3</b></td>
      <td><b>31.9</b></td><td><b>47.0</b></td><td><b>33.5</b></td>
      <td><b>25.5</b></td><td><b>36.1</b></td><td><b>26.3</b></td>
    </tr>
    <tr>
      <td>YOLO12s</td>
      <td>8.5</td><td>14.9</td><td>7.9</td>
      <td>26.0</td><td>38.7</td><td>27.6</td>
      <td>20.4</td><td>30.8</td><td>19.1</td>
    </tr>
    <tr>
      <td>YOLO12s + YCDa</td>
      <td><b>18.0</b></td><td><b>28.7</b></td><td><b>18.2</b></td>
      <td><b>33.7</b></td><td><b>48.6</b></td><td><b>36.6</b></td>
      <td><b>26.0</b></td><td><b>35.3</b></td><td><b>27.7</b></td>
    </tr>
  </tbody>
</table>

</div>


## Quick Start

```bash
cd YCDa/
python -m venv YCDa
source YCDa/bin/activate
pip install -e .
```

## Training

### YCDa-12s Model Pretrain
```bash
yolo detect train model=yolo12s-YCDa.yaml data=coco.yaml epochs=60 batch=32  device=0
```

### Camouflage task
```bash
yolo detect train model= data=yolo12s-YCDa.yaml epochs=300 pretrained=YCDa-12s-cocoPretrain.pt batch=16 patience=50 device=0
```

## Validation

```bash
yolo detect val model=YCDa-12s-COD10K-D.pt data=COD10K-D.yaml device=0 split=test
```

## Checkpoints

🔗 **[Download YCDa-12s-COD10K-D Checkpoints]()**


## Citation

If you find YCDa useful in your research, please consider citing our work.
