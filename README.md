<h1 align="center">Custom Road Signs Detector with YOLOv5 </h1>
<div align="center">
  A baby step to building a full-fledged Tesla 😝
</div>
<br>
<p align="center">
  <img src="https://github.com/ArdaniahJ/Custom_Object_Detection_with_YOLOv5/blob/main/YOLOv5.gif" alt="BeanTesla" />
</p>



This notebook explains custom training of YOLOv5 model implemented in PyTorch for object detection in Colab using the dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection). The YOLOv5 code is based on the official code from [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics. 

## About the Dataset
YOLO format is a text file that contains the information about the images and the objects in them. 
It uses a __.txt__ format for annotation. Each row in the file represents an annotation for an image.<br>
1. The first column contains the class label
2. The remaining columns contain the bounding box coordinates and object confidence scores. 

Below is an example of how the annotation data may look like in a YOLO .txt file:
```
0 0.42 0.31 0.23 0.56
1 0.62 0.56 0.25 0.38
2 0.14 0.84 0.17 0.28
```
In the above example, there are 3 annotations for the image;
1. __The first annotation__ has a;
  + class label of 0
  + bounding box with coordinates (0.42, 0.31) for the top-left corner
  + (0.23, 0.56) for width and height
2. __The second and third annotations__ have class lavel 1 and 2, respectively.
<br> Below is the overall format for the annotation:
```
[Class label] [center in X] [center in y] [Width] [Height]
```
<br>
However, the dataset is in __PASCAL VOC.xml format__ which is not the YOLOv5 format for training and testing. It is used for annotation, not for training and testing, thus said from YOLOv3 & YOLOv4. Therefore, there are __3 steps__ needed to convert the __.xml to .txt__ format.
<br>
1. Step 1: Parse the XML file 

```python
# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict
  ```

2. Step 2: Connvert the XML file
```python
# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"trafficlight": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join("annotations", info_dict["filename"].replace("png", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
```

3. Step 3: Get the annotations
```python
# Get the annotations
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]
annotations.sort()

# Convert and save the annotations
for ann in tqdm(annotations):
    info_dict = extract_info_from_xml(ann)
    convert_to_yolov5(info_dict)
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]
```

Below is the image as the result after testing the converted annotation <br>

![Annotation testing](https://github.com/ArdaniahJ/Custom_Road_Signs_Detector_with_YOLOv5/assets/120354757/5f39a40c-acfd-4c4b-96ff-2e580049c01b)

Road Signs Dataset are grouped into four classes:
1. Traffic Light
2. Stop
3. Speedlimit
4. Crosswalk


## Project Code 

The notebook can be viewed publicly on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kG-WgnZCJaeFDR9clfYgWDAyT34SPQNG?usp=sharing)


# Setup the YAML config File for Training 
This is the heart of the project where the `data.yaml` is customly configured. YAML config file is used to specify the configuration settings for YOLO model. It contains various parameters and options that define the architecture, training settings, and other configurations of the model that relates to the dataset. The following parameters are to be defined:
1. `train`, `test`, & `val`: Location of the train, test and validation images. 
2. `nc`: Number of classes in the dataset
3. `names`: Names of the calsses in the dataset. The index of the classes in this list would be used as an identifier for the class names in the code. 

Create a new file called `road_sign_data.yaml` and place it in the `yolov5/data` folder. Then populate it with the foloowing. 
``` python
train: /content/yolov5/Road_Sign_Dataset/images/train/
val: /content/yolov5/Road_Sign_Dataset/images/val/
test: /content/yolov5/Road_Sign_Dataset/images/test/

# number of classes
nc: 4

# class names
names: ["trafficlight","stop", "speedlimit","crosswalk"]
```
YOLOv5 expects to find the training labels for the images in the folder whose name can be derived by replacing `images` with `labels` in the path to dataset images. For example, in the above example, YOLOv5 will look for train labels in `/content/yolov5/Road_Sign_Dataset/images/train/`.

# Hyperparameter Config File
This config file on the other hand helps to define the hyperparameter for NN during the training process. This project used the default config which is `data/hyp.scratch-low.yaml`. 

# YOLOv5 Custom Network Architecture 
`yolov5s.yaml` contains the architecture required for training on COCO dataset. However, since this project doesn't use COCO dataset, the configuration thus needs to be changed. Below is the magic command `writetemplate` that can be used in IPython to write the contents of a code block to a specific file, in this case; the hyperparameter needed to be changed is only number of classes from `nc: 80` to `nc: 4`.
```python
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
```
# Training the Custom Road Signs Detector Model
The YOLOv5s model is trained by specifying __dataset, batch-size, image size & pretrained weights yolov5s.pt__. Parameters used are as below:
+ `--img` - specifies the input image size as 640x640 pixels
+ `--batch` - batch size (model weights are updated with each batch)
+ `--epochs` - number of trainings to run
+ `--cfg` - path to the yolov5s network architecture
+ `--hyp` - path to the hyp.scratch-low.yaml which contains the training hyperparameters
+ `--data` - path to the road_sign_data.yaml file (that relates to the dataset)
+ `--weights` - path to the initial weights file to use for the model
+ `--workers` - sets the number of data loading workers to speed up the training process
+ `--name` - name of the training run, which will be used to create a directory to store the training results

# Detector Performance Log
There are two ways of keeping in track of the model's training performance.
1. __Remotely__: Tensorboard
  + Training:![tensorboard remote log 1](https://github.com/ArdaniahJ/Custom_Road_Signs_Detector_with_YOLOv5/assets/120354757/3ed63982-4baa-48fd-aab7-e869e2bc3156)
  + Testing: ![tensorboard remote log 2](https://github.com/ArdaniahJ/Custom_Road_Signs_Detector_with_YOLOv5/assets/120354757/e411a0d8-f00d-45e0-982e-2423b5a79dc0)
2. __Locally__: Some old school graph in case tensorboard isn't working 😛
![detector local log graph](https://github.com/ArdaniahJ/Custom_Road_Signs_Detector_with_YOLOv5/assets/120354757/a0d80168-55b1-41e8-87d8-66ffa45d44b8)


## Detection Example
![sample of detection](https://user-images.githubusercontent.com/120354757/207774813-962ea140-bf1c-4880-989c-fb509121ffa1.png)

# Inference with custom YOLOv5s Trained Weights
Inference is the deployment phase of the object detection model (in this case, the road signs detector) on new, unseen images or videos to detect and locate objects. 
During inference, the trained YOLO model takes an input image or video frame and processes it through its layer to generate bounding box predictions and class probabilities for the detected objects. These predictions are then used to identify and locate objects in the input data. The parameters used are breakdown below:
```
%cd /content/yolov5/: This line changes the current directory to /content/yolov5/. The %cd command is typically used in Jupyter Notebook or Google Colab to change the working directory.

!python detect.py --source './Road_Sign_Dataset/images/test' --weights './runs/train/yolo_road_det/weights/best.pt' --conf 0.25 --name yolo_road_det: This line executes a Python script named detect.py with the following command-line arguments:

--source './Road_Sign_Dataset/images/test': Specifies the source path for the input images to be detected. In this case, it is the ./Road_Sign_Dataset/images/test directory.

--weights './runs/train/yolo_road_det/weights/best.pt': Specifies the path to the weights file of the trained model. The weights file best.pt is located in the ./runs/train/yolo_road_det/weights/ directory.

--conf 0.25: Sets the confidence threshold for object detection to 0.25. This threshold determines the minimum confidence score required for a detected object to be considered valid.

--name yolo_road_det: Specifies the name or identifier for this inference run. It can be used for saving the output or for tracking different runs.

The command runs the detect.py script, which performs object detection on the specified source images using the trained YOLO model. The output of the detection process will depend on the implementation of detect.py and may include bounding box coordinates, object labels, confidence scores, and visualized results.
```

