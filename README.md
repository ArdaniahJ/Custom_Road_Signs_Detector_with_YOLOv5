# Object Detection with Custom Dataset using YOLOv5 


<p align="center">
  <img src="https://github.com/ArdaniahJ/MrBeanYOLOv5/blob/main/Object%20Detection%20on%20the%20road.gif" alt="BeanTesla" />
</p>



This notebook explains custom training of YOLOv5 model implemented in PyTorch for object detection in Colab using the dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection). The YOLOv5 code is based on the official code from [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics. 

## About the Dataset
The dataset is in __.xml format__.


### Project Code [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kG-WgnZCJaeFDR9clfYgWDAyT34SPQNG?usp=sharing)

## Requirements

* Local machine
  1. Python 3.9 or higher (v3.9 is the version I used while doing this project)
  2. OpenCV
  3. PyTorch 11.2 or higher

* Web IDE
  1. Google Colab 
  2. JupyterLab
  
## Detection Example

![sample of detection](https://user-images.githubusercontent.com/120354757/207774813-962ea140-bf1c-4880-989c-fb509121ffa1.png)

## Running the detector

### On a single or multiple images
Clone, and `cd` into the repo directory.
  * The first thing you need to do is to get the weights file. This time around for v5, authors has supplied the weight files only for COCO dataset here and placed the weight files into the YOLOv5 repo directory. 
  * Or, you could just write the code for the process.
  ```python
  wget https://pjreddie.com/media/files/yolov3.weights 
  python detect.py --images imgs --det det 
  ```
  The flags in the code above is explained as below;
  * ```--images```flag defines the directory to load images from, or a single image file (it'll figure it out)
  * ```--det``` is the directory to save images to
  * ```--bs``` is the batch size 
  
  Object threshold confidence can be tweaked with flags that can be looked up with;
  ``` 
  python detect.py -h 
  ```
  
 ### Speed Accuracy Tradeoff
  * ```--reso``` glag is to change the resolution of the input images.
  * The default value is 416. Whatever value you chose, remember it should be a x32 or >32.
  * Weird things will happen if you don't. 
  * You've been warned ✋
  
  
