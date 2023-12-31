# effnet
# Changes
* January 1, 2024
  * args in nas.py and automated some stuffs
  * mixed the datasets so just check below 
  * manually augmented the dataset
      * 4 random rotation from -360~360 degrees
      * motion blurred with k=15
      * flipped the images
  * etc. 
# before all of the installation and other things:

- gpu is required, i dont know if you can train in cpu while having smaller batch size, but i think its possible
- if going to use gpu, cuda and cudnn is required, i recommend using docker/wsl/linux os for easy installation of cuda and cudnn
- make sure to check that the compatibility of cuda and torch are compatible, https://pytorch.org/get-started/previous-versions/, just check the system logs for the gpu, python version, pip list, and cuda and cudnn version
- [google colab notebook here](https://colab.research.google.com/drive/1RrgsdnAgvHFcr9lQhsYEdZk6Kz_lFTWE?usp=sharing)

### python = 3.8

# Run these command first:

- git clone https://github.com/laefy13/buh.git
- cd buh
- git clone https://github.com/laefy13/pytorch-image-models.git
- git clone https://github.com/laefy13/keras-efficientnets.git
- git clone https://github.com/laefy13/WSTMD.git

### for installing

- pip install -r req_docker.txt
- cd keras-efficientnets
- pip install .
- cd ../WSTMD
- pip install -r requirements.txt
- cd ..

### Dataset

- [Download Here](https://drive.google.com/drive/folders/16VkD09XaSQHnmRi00WGHFLxg15uYbVhh?usp=sharing)
- put the images_2 folder in the buh folder so the directory will look like:
  buh
   |--images_2
   |--logs
   |--output_images
   |--other folders

### Needed to be download for WSTMD

- https://www.kaggle.com/datasets/clearhanhui/biyesheji for the dataset,
  - just move all of the image to the the WSTMD/data/img or
  - rename the folder to tongue then use the move_THIS.py, not tested but should work
- https://download.pytorch.org/models/resnet34-333f7ec4.pth for the pre_weight, and move this insed WSTMD folder

### To run nas? and get logs for evaulation:

- python nas_question_mark.py

### system info logs

- you can run the system_getter.py for refernce on what system you are using for training and whatnot

### if having problems installing or want to install in other ways:

- you can also just refer to some of the system info logs in logs folder, the system info logs contain the gpu, python version, pip list, and cuda and cudnn version
- if going to use docker/vast.ai this is the image/template i used for the first test: pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# HYPERPARAMETERS ON THE NAS?

- ermm batch size is 32 on the original effnet paper and the batch size is also 32 on the resnet34, so the plan is to go for 32 batch size but because of vram consumption in while running the nas? I temporarily set that to 16, since a 3060 with 12GB VRAM cant handle 32 batch size.
- the possible things to do regarding this is:
  - somehow make 32 batch size run in 12GB vram, making the script sleep + clean the pytorch things after training to make sure that the GPU is clean
  - just get better gpu, the one used in the resnet34 used a gpu that has 20gb vram

# things that will probably be added/updated

- better optimization
- better hyperparameters
- better nas?
- tongue segmentation with deeplab
- making the logs cleaner for easy getting of the info
- etc.
