# dataset settings
dataset_type = 'CocoDataset'  # Tells MMDetection to use the COCO format (JSON-based annotation).
data_root = '/home/woody/iwi5/iwi5237h/VinDr-CXR/'      # path to your dataset (images and annotation files).
classes = ("Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Consolidation","ILD","Infiltration","Lung Opacity","Nodule/Mass","Other lesion","Pleural effusion","Pleural thickening","Pneumothorax","Pulmonary fibrosis","No finding")
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None         # Used for loading from non-local sources like S3, HTTP, etc.

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
        # NEW: Per-class oversampling (applies BEFORE augmentation)
    dict(
    type='RandomSampler',
    # Oversample rare classes (Pneumothorax, Consolidation etc.)
            sample_ratio = {
        # Ultra-rare classes (<1%): Aggressive oversampling
        'Pneumothorax': 15.0,      # 0.31% → 4.65% (163→2445 samples)
        'Consolidation': 12.0,      # 0.72% → 8.64% (376→4512)
       #       # Assuming exists (else remove)
        
        # Rare classes (1-2%): Moderate oversampling
        'Calcification': 8.0,       # 1.31% → 10.48% (688→5504)
        'ILD': 7.0,                 # 1.49% → 10.43% (782→5474)
        'Infiltration': 6.0,        # 1.75% → 10.5% (920→5520)
        
        # Medium-frequency (3-5%): Slight boost
        'Nodule/Mass': 2.5,         # 3.82% → 9.55% (2004→5010)
        'Other lesion': 2.0,        # 3.07% → 6.14% (1611→3222)
        'Lung Opacity': 1.8,        # 3.46% → 6.23% (1816→3269)
        'Pleural effusion': 1.5,    # 3.45% → 5.18% (1810→2715)
        
        # Common classes: Maintain or reduce
        'Pleural thickening': 1.0,  # 6.86% → 6.86% (no change)
        'Pulmonary fibrosis': 0.8,   # 6.38% → 5.1% (reduce slightly)
        'Cardiomegaly': 0.7,        # 8.16% → 5.71% (reduce)
        'Aortic enlargement': 0.6,   # 10.72% → 6.43% (reduce)
        
        # Dominant class: Drastically reduce
        'No finding': 0.1           # 48.5% → 4.85% (25467→2547)
    }
      ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=15),  # Rotation
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    # _delete_=True,
    batch_size=3,
    num_workers=6,   # Use 2 CPU workers for loading data more quickly.
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  # controls how data is selected from the dataset for each batch.
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    # sampler=dict(type='ClassBalancedSampler', oversample_thr=1e-3), 
    # batch_sampler=dict(type='AspectRatioBatchSampler'),

    # sampler=dict(type='ClassAwareSampler'),    # Balances class frequency in batches	Doesn't create new samples
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data1/train1.json',
        data_prefix=dict(img='data1/train1/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args))
        
val_dataloader = dict(
    # _delete_=True,

    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,         # Keep the last batch even if it’s smaller than batch_size.
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data1/val1.json',
        data_prefix=dict(img='data1/val1/'),
        test_mode=True,      #  disables shuffling and some training-only behaviors.
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'data1/val1.json',
    metric='bbox',
    classwise=True,
    format_only=False,     # it just formats the output into a file (usually .json) for submission.
    backend_args=backend_args)


# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'vindrcxr_test.json',
        data_prefix=dict(img='vindrcxr_test/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    classwise=True,
    format_only=False,
    ann_file=data_root + 'vindrcxr_test.json',
     backend_args=backend_args
    # outfile_prefix='./work_dirs/coco_detection/test'
    )