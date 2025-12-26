
# dataset settings
dataset_type = 'ISPRSDataset'
data_root = '../data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
Pots_IRRG_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
Vail_IRRG_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='ISPRSDataset',
            data_root='../data/potsdam/train_src',
            img_dir='all_imgs',
            ann_dir='all_labels',
            pipeline=Pots_IRRG_train_pipeline),
        target=dict(
            type='ISPRSDataset',
            data_root='../data/vaihingen/train_src',
            img_dir='all_imgs',
            ann_dir='all_labels',
            pipeline=Vail_IRRG_train_pipeline)),
    val=dict(
        type='ISPRSDataset',
        data_root='../data/vaihingen_val',
        img_dir='all_imgs',
        ann_dir='all_labels',
        pipeline=test_pipeline),
    test=dict(
        type='ISPRSDataset',
        data_root='../data/vaihingen_test',
        img_dir='all_imgs',
        ann_dir='all_labels',
        test_mode=True,
        pipeline=test_pipeline))
