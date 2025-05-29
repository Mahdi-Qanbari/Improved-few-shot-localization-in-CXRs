
pip install -e . -v MMCV_WITH_OPS=1 

# batch processing
sbatch.tinygpu infer_job.sh
rm inference_output_912350.log

# remove logs
rm inference_*.log


 conda install pytorch==2.0.1 torchvision==0.15.2  pytorch-cuda=11.8 -c pytorch -c nvidia

import torch
print(torch.cuda.is_available())   # Should return True

pip uninstall torch torchvision 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124


import torch
x = torch.rand(5, 3).cuda()
print(x)


conda env remove -n openmmlab 




python -c "
from mmdet.apis import DetInferencer
import torch
print('Downloading model weights...')

# Use DetInferencer to trigger the download
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device='cuda:0')

# Check if the model weights are in the cache directory
model_cache_dir = torch.hub.get_dir()
print(f'Models should be stored in: {model_cache_dir}')
"
models = DetInferencer.list_models('mmdet')


pip install mmcv-full==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.4.1/index.html

# copying from local machine to remote
scp -r /home/mahdi/Downloads/pytorch_model.bin csnhr.nhr.fau.de:/home/hpc/iwi5/iwi5237h/bert-base-uncased

scp -r  ./COCO_val2017.zip csnhr.nhr.fau.de:/home/woody/iwi5/iwi5237h


#The environment from the calling shell, like loaded modules, will be inherited by the interactive job.
# cuDNN is installed on all nodes and loading a module is not required.
# interactive job
 salloc.tinygpu --gres=gpu:1 --time=01:00:00
 module load python/3.12-conda
 module load cuda/12.4.1
 conda activate openmmlab
 python coco_inference.py | tee output.log


cd My_Pytorch/src_to_implement
cd Pytorch_project1/src_to_implement
python PytorchChallengeTests.py TestDataset

# List installed packages :
conda list



####################################
#pytorch
python PytorchChallengeTests.py TestDataset



# Copy to local
scp -r csnhr.nhr.fau.de:Pytorch_project1/src_to_implement ~/backup

# seeing Quotas
shownicerquota.pl

#Show all output from my_script.py on your screen And write the same output to output.log
python my_script.py | tee output.log



