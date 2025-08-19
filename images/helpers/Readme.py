#------------------ Training the base model ---------------------

# 1 num_classes in the model head:
        num_classes=14,  # Change to your actual number of classes
#Match mean and std in data_preprocessor to your dataset (if not using ImageNet).




# 2. Apply Class-Balanced Sampler:
train_dataloader = dict(
    sampler=dict(
        type='ClassBalancedSampler',
        oversample_thr=0.1  # Tune this (e.g., 0.05–0.5)
    ),
    ...
)



# 3. Use Weighted Loss or Focal Loss (optional)

#  Recommendation for you (ViDR-CXR, 15 classes, imbalanced):

#     ➤ Use ClassBalancedSampler + Focal Loss,
#     or
#     ➤ Use ClassBalancedSampler + Weighted Cross Entropy if you prefer stability and have frequency stats.
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[w1, w2, ..., w15],  # Based on your class frequency
            loss_weight=1.0
        )
    )
)


#OR switch to FocalLoss (to down-weight easy examples and focus on hard ones):
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0
)
#⚠ If switching to FocalLoss, use_sigmoid must be True.


# 4. Dataset class list & labels

#Make sure your metainfo is defined correctly:   ------> 14 
metainfo = dict(
    classes=("Aortic enlargement", "Atelectasis", ..., "No finding")
)

#5. Optional: Data Augmentation (to prevent overfitting oversampled rare images)

#In your train_pipeline, consider using:
dict(type='RandomFlip', prob=0.5),
dict(type='RandomCrop', crop_size=(512, 512)),
dict(type='ColorJitter', brightness=0.2, contrast=0.2),
...



#  6. Logging + Evaluation Strategy

# Make sure you evaluate on both:

#     Full set (including "No finding")

#     Subset (excluding "No finding")

# You can do this by setting up two val_evaluator configs or applying filters during analysis.

