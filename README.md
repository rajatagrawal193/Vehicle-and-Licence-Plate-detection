# Vehicle Detection, Tracking and Re-Identification along with Number Plate Detection and Recognition

The implemetation uses YoloV3 from darknet framework, pytorch, tensorflow-gpu and keras 

# Demo

## Installation Steps

## Requirements: CUDA 10.0, Cudnn >=7.6, NCCL ==2.2.1 (For GPU)
    
1. Install darknet:
    
    `cd darknet && make clean
    make all
    cp -f libdarknet.so ../.
    cd ..`
    
2. Create an anaconda environment, using `conda create -n <env_name>`. If you don't have anaconda installed, [install anaconda](https://docs.anaconda.com/anaconda/install/linux/)

3. Activate anaconda environment using `conda activate <env_name>`

4. Run `source install.txt` will install all the necessary conda packages. Just press `y` whenever prompted

## Run Demo

`python main.py /path/to/input/video/file`

Output will be saved in `output/<input_name>_result_tracker<input_img_size>.avi` . The detected number plates will be stored at `output/license_plates` . The detected cars will be stored at `output/cars` . Average FPS will be printed on the console after output has been saved.

`python main.py -h` for more options
