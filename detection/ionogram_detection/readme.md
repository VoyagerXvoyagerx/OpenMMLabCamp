# 基于YOLO系列算法的频高图度量benchmark

## 数据集构建

电离层频高图是获取电离层实时信息最重要的途径。电离层不规则结构变化特征研究对检测电离层不规则结构，精准提取和度量电离层各层轨迹和关键参数，具有非常重要的研究意义。

利用中国科学院在海南、武汉、怀来获取的不同季节的4311张频高图建立数据集，人工标注出E层、Es-c层、Es-l层、F1层、F2层、Spread F层共6种结构。[download](https://drive.google.com/file/d/1MZUonB6E0o7lq_NndI-F3PEVkQH3C8pz/view?usp=sharing)

File tree:

```shell
Iono4311/
├── annotations
│   ├── annotations_all.json
│   ├── class_with_id.txt
│   ├── test.json
│   ├── train.json
│   └── val.json
├── classes_with_id.txt
├── dataset_analysis.ipynb
├── dataset.ipynb
├── images
├── labels
├── test_images
├── train_images
└── val_images
```

数据预览

![标注好的图像示例](https://github.com/VoyagerXvoyagerx/OpenMMLabCamp/blob/main/detection/ionogram_detection/20130401070700.jpg "fig1")

1. 使用MMYOLO提供的脚本将 labelme 的 label 转换为 COCO label

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ./Iono4311/images --labels-dir ./Iono4311/labels --out ./Iono4311/annotations/annotations_all.json
```

2. 使用下面的命令可以将 COCO 的 label 在图片上进行显示，这一步可以验证刚刚转换是否有问题

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ./Iono4311/images --ann-file ./Iono4311/annotations/annotations_all.json
```

3. 划分训练集、验证集、测试集，设置70%的图片为训练集，15%作为验证集，15%为测试集

```shell
python tools/misc/coco_split.py --json ./Iono4311/annotations/annotations_all.json \
                                --out-dir ./Iono4311/annotations \
                                --ratios 0.7 0.15 0.15 \
                                --shuffle \
                                --seed 14
```

数据集中各类别实例数量，notebook代码在[/tools/dataset_analysis.ipynb](OpenMMLabCamp/detection/ionogram_detection/tools/dataset_analysis.ipynb)

```python
annotations_all 4311 images
      E  Esl  Esc    F1    F2  Fspread
0  2040  753  893  2059  4177      133

train 3019 images
      E  Esl  Esc    F1    F2  Fspread
0  1436  529  629  1459  2928       91

val 646 images
     E  Esl  Esc   F1   F2  Fspread
0  311  101  137  303  626       20

test 646 images
     E  Esl  Esc   F1   F2  Fspread
0  293  123  127  297  623       22
```

4. 配置文件

配置文件在路径/config/custom_dataset下

5. 数据集可视化分析

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-50e_ionogram.py \
                                                --output-dir output
```

![可视化分析](/detection/ionogram_detection/Dataset_bbox_area.jpg)

E、Es-l、Esc、F1类别以小目标居多，F2、Fspread类主要是中等大小目标。

6. 可视化config中的数据处理部分

```shell
python tools/analysis_tools/browse_dataset.py configs/custom_dataset/yolov5_m-v61_syncbn_fast_1xb32-50e_ionogram.py \
--output-dir output --show-interval 1
```

7. 修改Anchor尺寸

```shell
python tools/analysis_tools/optimize_anchors.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram.py \
                                                --algorithm v5-k-means \
                                                --input-shape 640 640 \
                                                --prior-match-thr 4.0 \
                                                --out-dir work_dirs/dataset_analysis_5_s
```

8. 训练

```shell
python tools/train.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram_pre0.py
```

9. 测试

```shell
python tools/test.py ./configs/custom_dataset/yolov6_l_syncbn_fast_1xb32-100e_ionogram.py \
./work_dirs/yolov6_l_100e/best_coco/bbox_mAP_epoch_76.pth \
--work-dir ./work_dirs/yolov6_l_100e \
--show-dir ./output \
--wait-time 1
```

10. 测试模型Param、FLOPs

[参考脚本](https://github.com/open-mmlab/mmyolo/blob/2875d8b64e75b34c2a7f4cf134f9348c2f018ed9/tools/analysis_tools/get_flops.py) 一个没有被 merge 的 pr

编写一键打印所有模型的notebook[get_flops.ipynb](/detection/ionogram_detection/tools/get_flops.ipynb)

## 实验结果

| Model | epoch(best) | FLOPs(G) | Params(M) | pretrain | val mAP | test mAP | config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5-s | 50(50) | 7.95 | 7.04 | Coco | 0.579 |  | yolov5_s-v61_syncbn_fast_1xb32-50e_ionogram |
| YOLOv5-s | 100(75) | 7.95 | 7.04 | Coco | 0.577  |  | yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram |
| YOLOv5-s | 200(145) | 7.95 | 7.04 | None | 0.565 |  | yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram_pre0 |
| YOLOv5-m | 100(70) | 24.05 | 20.89 | Coco | 0.587  | 0.586 | yolov5_m-v61_syncbn_fast_1xb32-100e_ionogram |
| YOLOv6-s | 100(54) | 24.2 | 18.84 | Coco | 0.584 |  | yolov6_s_syncbn_fast_1xb32-100e_ionogram |
| YOLOv6-s | 200(188) | 24.2 | 18.84 | None | 0.557 |  | yolov6_s_syncbn_fast_1xb32-100e_ionogram_pre0 |
| YOLOv6-m | 100(76) | 37.08 | 44.42 | Coco | 0.590 |  | yolov6_m_syncbn_fast_1xb32-100e_ionogram |
| YOLOv6-l | 100(76) | 71.33 | 58.47 | Coco | 0.605 | 0.597 | yolov6_l_syncbn_fast_1xb32-100e_ionogram |
| YOLOv7-l | 100(88) | 52.42 | 37.22 | Coco | 0.590 |  | yolov7_l_syncbn_fast_1xb32-100e_ionogram |
| YOLOv7-x | 100(58) | 94.27 | 70.85 | Coco | 0.602 |  | yolov7_x_syncbn_fast_1xb32-100e_ionogram |
| rtmdet-l | 100(80) | 79.96 | 52.26 | Coco | 0.601 |  | rtmdet_l_syncbn_fast_1xb32-100e_ionogram |
| rtmdet-x | 100(94) | 141 | 94.79 | Coco | 0.603 |  | rtmdet_x_syncbn_fast_1xb32-100e_ionogram |

[训练过程可视化](https://wandb.ai/19211416/mmyolo-tools/reports/Object-Detection-for-Ionogram-Automatic-Scaling--VmlldzozNTI4NTk5)

现有的实验结果中，YOLOv6-l的验证集mAP最高。

对比loss下降的过程可以发现，使用预训练权重时，loss下降得更快。可见即使是自然图像数据集上预训练的模型，在雷达图像数据集上微调，也可以加快收敛

![loss](/detection/ionogram_detection/loss.png)

## 自定义数据集config修改经验

### 必须修改的项目

- \_base\_
- work_dir

### 模型尺寸不变，修改策略时

继承自修改过的config
根据实验需要修改config内容

### 修改模型尺寸时

继承自修改过的config

- num_classes related (e.g. loss_cls)
- load_from
- 官方config中的内容

### 使用新的模型训练自定义数据集

继承自官方config

- visualizer
- dataset settings
  - data_root
  - class_name
  - num_classes
  - metainfo
  - img_scale
- train, val, test
  - batch_size, num_workers
  - train_cfg
    - max_epochs, save_epoch_intervals, val_begin
  - default_hooks
    - max_keep_ckpts
    - save_best
  - lr
  - val_dataloder, test_dataloader
    - metainfo
    - root
  - val_evaluator, test_evaluator

## To Do

- 完善测试内容
- 使用两阶段模型
- 改进模型
