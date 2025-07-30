default_scope = 'mmdet'

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet_custom.hooks.log_classwise_map_hook',
    ])


    
default_hooks = dict(
    timer=dict(type='IterTimerHook'),                   # Measures iteration time — useful for profiling/training speed.
    logger=dict(type='LoggerHook', interval=100),        #Logs losses, metrics, and info every interval steps (here: every 50 iterations = batches).
    # loggers = [
    # dict(type='TextLoggerHook'),
    # dict(type='TensorboardLoggerHook')
    # ],

    param_scheduler=dict(type='ParamSchedulerHook'),    #Applies the learning rate (LR) schedule.
    checkpoint=dict(type='CheckpointHook', interval=10 , save_best='coco/bbox_mAP', rule='greater' ), #Saves model checkpoints every interval epochs (here: every epoch).
    sampler_seed=dict(type='DistSamplerSeedHook'),      #Makes sure the data loader shuffles differently each epoch — useful in distributed training.
    visualization=dict(type='DetVisualizationHook')) 
       #Draws predictions on images (if enabled), can be used for monitoring model outputs.
param_scheduler = [
    dict(type='StepLR', step_size=10, gamma=0.8, by_epoch=True)
]

#

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',  # or another validation metric like 'val_loss'
        rule='greater',            # 'greater' = higher is better, 'less' = lower is better
        patience=5,                # stop if no improvement in 5 eval intervals
        min_delta=0.005              # minimum change to qualify as improvement
    ),
    dict(
        type='LogClasswiseMAPHook',
        csv_path='work_dirs/classwise_map_log.csv'
    ),
]





env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='TensorboardVisBackend', save_dir='runs')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)  #Averages metrics over 50 iterations per epoch

log_level = 'INFO'
load_from = None #'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'        # Load pretrained weights for transfer learning
resume = False         #	it is set in the batc file train_detr.slurm