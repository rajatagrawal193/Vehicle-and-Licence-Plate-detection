The implemetation uses YoloV3 from darknet framework. 

1. Install darknet:
    
    cd darknet && make clean
    make all
    cd ..

2. pip install -r requirements.txt (All the programs have been tested on python 2.7)

3. python main.py /path/to/input/video/file

  Output will be saved at the location of the input with name as <input_name>_result_tracker.mp4
