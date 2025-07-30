#!/bin/bash -l
#SBATCH --job-name=detr
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1      
#SBATCH --gres=gpu:rtx3080:2
#SBATCH --cpus-per-task=6 
#SBATCH --time=00:30:00
#SBATCH --partition=rtx3080

module load python/3.12-conda
module load cuda/12.4.1
conda activate openmmlab
export PYTHONPATH=/home/hpc/iwi5/iwi5237h/Few_Shot_DETR:$PYTHONPATH


CONFIG=detr_r50_8xb2-150e_coco.py
COCO_INIT=checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth
WORK_DIR=work_dirs/detr_r50_8xb2-150e_coco
RESUME_CHECKPOINT=$(cat $WORK_DIR/last_checkpoint)

echo "Starting distributed training on 2 GPUs..."

if [ -f "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from latest checkpoint: $RESUME_CHECKPOINT"
    srun python -m torch.distributed.run --nproc_per_node=2 tools/train.py $CONFIG --resume --launcher pytorch
else
    echo "Starting fresh training using COCO pre-trained weights"
    srun python -m torch.distributed.run --nproc_per_node=2 tools/train.py $CONFIG --launcher pytorch \
        --cfg-options load_from=$COCO_INIT
fi
