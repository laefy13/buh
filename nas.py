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
import argparse
import os
import subprocess
import time
from WSTMD.get_flops import wstmd_flops
import yaml
import torch

def _parse_args():
    # Do we have a config file to parse?
    args = parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--output-dir', metavar='DIR',
                    help='path to output')
parser.add_argument('--in-dir', metavar='DIR',
                    help='path to output')
                    
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
parser.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
# get flops
if not os.path.exists('sys_flops.txt'):
    curr_flops = wstmd_flops()
    with open('sys_flops.txt','w') as f:
        f.write(str(curr_flops))
else:
    with open('sys_flops.txt','r') as f:
        curr_flops=float(f.readlines()[0])
        
        
# function for producing an array of an array of coefficients, depth, width and resolution


if not os.path.exists('coefficients.txt'):
    from keras_efficientnets.optimize import optimize_coefficients
    from keras_efficientnets.optimize import get_compound_coeff_func
    results = optimize_coefficients(phi=1., max_cost=2.0, search_per_coeff=10)
    with open('coefficients.txt', 'w') as coff:
        for i in range(len(results)):
            write_this =f'{results[i][0]},{results[i][1]},{results[i][2]}\n'
            coff.write(write_this)

else:
    with open('coefficients.txt','r') as f:
        results=f.readlines()  
        
#loop that will executre the train.py of the pytorch-image-models repo

args, args_text = _parse_args()
trained_len = len(results)
existing_combinations = set()
if os.path.exists(f'{args.in_dir}/trained.txt'):
  with open(f'{args.in_dir}/trained.txt','r') as tr:
      tr.seek(0)
      for line in tr:
          existing_combinations.add(tuple(map(float, line.strip().split(','))))

run_this = f"python -u pytorch-image-models/train.py \
            --epochs 5 \
            --log-interval 1 \
            --data-dir './images_2/' \
            --model efficientnet_b0 \
            --opt rmsprop \
            --momentum .9 \
            --weight-decay 1e-5 \
            --decay-epochs 2.4 \
            --decay-rate .97 \
            --drop .2 \
            --num-classes 2 \
            --flops {curr_flops}\
            --no-aug \
            --output {args.output_dir}"
if args.lr_base:
  run_this+=f" --lr-base {args.lr_base}"
if args.lr:
  run_this+=f" --lr {args.lr}"
print(args.output_dir)
for i in range(len(results)):
    # if i < 30:
    #   continue
    tr = open(f'{args.output_dir}/trained.txt', 'a+')
    try:
        d,w,r = results[i].strip().split(',')
        depth = float(d)
        width =  float(w)
        resolution = round(224 *  float(r))
        if (depth, width, float(r)) in existing_combinations:
            tr.write(f'{depth},{width},{float(r)}\n')   

            continue
        batch_size = 32
        run_this+=f" --model-kwargs chann_mult={width} dep_mult={depth} \
            --input-size 3 {resolution} {resolution} \
            --batch-size {batch_size} \
            --validation-batch-size {batch_size} "
        print(run_this)
        subprocess.run(run_this,shell=True, check=True)
        tr.write(f'{depth},{width},{float(r)}\n')
        print('going to sleep zzzzzzzzzz')
        torch.cuda.empty_cache()
        time.sleep(50)
        torch.cuda.empty_cache()
    except Exception as e:
        continue

    finally:
        tr.close()