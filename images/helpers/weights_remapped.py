import torch

# --- CONFIGURATION ---
ckpt_path = "detr_14cls.pth"         # Path to the 14-class model
new_ckpt_path = "detr_15cls.pth"     # Output path for 15-class model

old_num_classes = 14
new_num_classes = 15

# --- LOAD CHECKPOINT ---
ckpt = torch.load(ckpt_path, map_location='cpu')

# DETR: classification head is usually here
cls_embed_key = 'model.bbox_head.cls_branches.0.weight'  # Check for your model key
bias_key = 'model.bbox_head.cls_branches.0.bias'

# --- UPDATE CLASS EMBEDDING ---
old_cls_weight = ckpt[cls_embed_key]     # [num_classes+1, hidden_dim]
old_cls_bias = ckpt[bias_key]

# Sanity check
assert old_cls_weight.shape[0] == old_num_classes + 1, "Old checkpoint class count mismatch"

# Expand weights to new size (15 classes + 1 background)
new_cls_weight = torch.nn.functional.pad(old_cls_weight, (0, 0, 0, 1))  # pad 1 row
new_cls_bias = torch.nn.functional.pad(old_cls_bias, (0, 1))            # pad 1 bias value

# Optionally, init last row (new class) with random values
torch.nn.init.normal_(new_cls_weight[-2], mean=0.0, std=0.01)  # index -2: new class
torch.nn.init.constant_(new_cls_bias[-2], 0.0)

# Replace in checkpoint
ckpt[cls_embed_key] = new_cls_weight
ckpt[bias_key] = new_cls_bias

# --- SAVE NEW CHECKPOINT ---
torch.save(ckpt, new_ckpt_path)

print(f"✅ Updated checkpoint saved to: {new_ckpt_path} with {new_num_classes} classes")
