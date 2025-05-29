default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),                   # Measures iteration time — useful for profiling/training speed.
    logger=dict(type='LoggerHook', interval=50),        #Logs losses, metrics, and info every interval steps (here: every 50 iterations = batches).
    param_scheduler=dict(type='ParamSchedulerHook'),    #Applies the learning rate (LR) schedule.
    checkpoint=dict(type='CheckpointHook', interval=1), #Saves model checkpoints every interval epochs (here: every epoch).
    sampler_seed=dict(type='DistSamplerSeedHook'),      #Makes sure the data loader shuffles differently each epoch — useful in distributed training.
    visualization=dict(type='DetVisualizationHook'))    #Draws predictions on images (if enabled), can be used for monitoring model outputs.

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)  #Averages metrics over 50 iterations per epoch

log_level = 'INFO'
load_from = None          # Load pretrained weights for transfer learning
resume = False            #	Resume training exactly where it left off (model + optimizer + scheduler)
