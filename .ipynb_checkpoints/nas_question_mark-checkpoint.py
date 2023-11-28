'''
This script is just for looping the train.py from the pytorch-image-models repo
and this is where you can adjust the hyperparameters for the effnetb0
if editing the structure of the effnet is needed go to the pytorch-image-models repo instead


    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

'''
import os
import subprocess
import time
from WSTMD.get_flops import wstmd_flops
'''
TRAINING SCRIPT, ADJUSTABLE ACCORDINGLY
 python train.py --data-dir "./img/" --class-map "./txt/class.txt" --num-classes 2
 --model efficientnet_b0 --input-size 3 224 224 --batch-size 32 --validation-batch-size 32 
 --opt rmsprop --momentum .9 --weight-decay 1e-5 --lr .256 --decay-epochs 2.4 --decay-rate .97

 DEFAULT SETTINGS, BASED FROM THE EFFNET PAPER:

 INPUT_CHANNELS = 32
 IMG_RES = 224,224
 LAYERS = [1,1,2,2,3,3,4,1,1]
 BASE_FLOPS = 129.432634496 GFLOPs IN 

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_19:24:38_PDT_2019
    Cuda compilation tools, release 10.2, V10.2.89

    torch                              1.10.0
    torchprofile                       0.0.4
    torchvision                        0.11.1
    python                             3.8.18

    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 530.30.02              Driver Version: 531.61       CUDA Version: 12.1     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA GeForce GTX 1650 Ti      On | 00000000:01:00.0 Off |                  N/A |
    | N/A   44C    P8                4W /  N/A|     73MiB /  4096MiB |      0%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+

    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |    0   N/A  N/A        66      G   /Xwayland                                 N/A      |
    |    0   N/A  N/A        84      G   /code                                     N/A      |
    |    0   N/A  N/A       123      G   /code                                     N/A      |
    +---------------------------------------------------------------------------------------+


 THE BASE MODEL WILL BE THE SAME AS THE EFFNET B0 AS MENTIONED IN OUR PAPER
 THE MODELS THAT WILL BE SCALED ARE EFFNET B1-7 
 WHERE B1-3 WILL/MUST(?) HAVE SMALLER FLOPS THAN THE BASE FLOPS
 WHILE B4 WILL HAVE ALMOST THE SAME FLOPS AS THE BASE FLOPS
 AND B5-7 WILL HAVE HIGHER FLOPS AS  THE BASE FLOPS

 

'''
# get flops
curr_flops = wstmd_flops()

# function for producing an array of an array of coefficients, depth, width and resolution
# check the readme of the keras-efficientnets for the syntax of this function
# have to move the imports here because for some reason im getting Segmentation fault (core dumped)
from keras_efficientnets.optimize import optimize_coefficients
from keras_efficientnets.optimize import get_compound_coeff_func
results = optimize_coefficients(phi=1., max_cost=2.0, search_per_coeff=10, verbose=True,sort_by_loss=True)
with open('coefficients.txt', 'w') as coff:
    for result in results:
        coff.write(str(result))
    

#loop that will executre the train.py of the pytorch-image-models repo
#the hyperparameters currently being used is according to the original efficientnet paper (i dotn know if i missed some)
for i in range(len(results)):
    depth = results[i][0]
    width = results[i][1]
    resolution = round(224 * results[i][2])
    batch_size = 32
    run_this = f"python -u pytorch-image-models/train.py \
    --epochs 50 \
    --log-interval 1 \
    --data-dir './img/' \
    --class-map './txt/class.txt' \
    --model efficientnet_b0 \
    --input-size 3 {resolution} {resolution} \
    --batch-size {batch_size} \
    --validation-batch-size {batch_size} \
    --opt rmsprop \
    --momentum .9 \
    --weight-decay 1e-5 \
    --lr .256 \
    --decay-epochs 2.4 \
    --decay-rate .97 \
    --model-kwargs chann_mult={width} dep_mult={depth} \
    --drop .2 \
    --num-classes 2 \
    --flops {curr_flops}"
    print(run_this)
    subprocess.run(run_this,shell=True, check=True)
    print('going to sleep zzzzzzzzzz')
    time.sleep(10)
    
    