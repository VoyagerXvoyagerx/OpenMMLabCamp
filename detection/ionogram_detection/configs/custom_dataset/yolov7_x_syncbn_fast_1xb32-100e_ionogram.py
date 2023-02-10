_base_ = './yolov7_l_syncbn_fast_1xb32-100e_ionogram.py'

'''
Must modify: _base_, load_from, work_dir, num_classes related(e.g. loss_cls)
_base_ = Coco Config: 
- visualizer
- dataset settings
    - data_root
    - class_name
    - num_classes
    - metainfo
    - img_scale
- train, val, test
    - train_cfg
        - max_epochs, save_epoch_intervals, val_begin
    - max_keep_ckpts
    - lr
    - test_dataloader
    - test_evaluator
'''
'''
data_root = './Iono4311'
class_name = ('E', 'Es-l', 'Es-c', 'F1', 'F2', 'Spread-F')
num_classes = len(class_name)
metainfo = dict(
    classes = class_name,
    palette = [(250, 165, 30), (120, 69, 125), (53, 125, 34), (0, 11, 123), (130, 20, 12), (120, 121, 80)])
img_scale = (400, 360)

# visualizer
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
'''

# work_dir and pre-train
load_from = './work_dirs/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth'
work_dir = './work_dirs/yolov7_x_100e'
anchors = [[(14, 11), (44, 7), (32, 20)], [(24, 67), (40, 83), (64, 108)], [(117, 118), (190, 92), (185, 142)]]
strides = _base_.strides


model = dict(
    backbone=dict(arch='X'),
    neck=dict(
        in_channels=[640, 1280, 1280],
        out_channels=[160, 320, 640],
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.4,
            block_ratio=0.4,
            num_blocks=3,
            num_convs_in_block=2),
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(in_channels=[320, 640, 1280]),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides)))
