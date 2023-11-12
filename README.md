# effnet
### python = 3.8
# Run these command first:
* git clone https://github.com/laefy13/pytorch-image-models.git
* git clone https://github.com/laefy13/keras-efficientnets.git
### for installing 
* pip install -r req_docker.txt
* you can also just refer to some of the system info logs in logs folder, the system info logs contain the gpu, python version, pip list, and cuda and cudnn version
* if going to use docker/vast.ai this is the image/template i used for the first test: pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
# HYPERPARAMETERS ON THE NAS?
* ermm batch size is 32 on the original effnet paper and the batch size is also 32 on the resnet34, so the plan is to go for 32 batch size but because of vram consumption in while running the nas? I temporarily set that to 16, since a 3060 with 12GB VRAM cant handle 32 batch size.
* the possible things to do regarding this is:
  ** somehow make 32 batch size run in 12GB vram, making the script sleep + clean the pytorch things after training to make sure that the GPU is clean
  ** just get better gpu, the one used in the resnet34 used a gpu that has 20gb vram
### To run nas? and get logs for evaulation:
* python nas_question_mark.py
  

