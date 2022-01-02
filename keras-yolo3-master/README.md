# YOLO3 (Detection, Training, and Evaluation) for the Deep Learning Project

This folder was originally cloned from https://github.com/experiencor/keras-yolo3 
We have modified some of the code (mainly fixed some errors regarding bounding box predictions and enhanced some of the classes)

## Weights:

As the weight files are too large to be uploaded to Github, they are uploaded at google docs. We provide the link here:
https://drive.google.com/drive/folders/1PXrWi1S8Lw4lxZYLtlw1jPdosTahHy1j?usp=sharing

Due to us performing active learning, there are multiple networks and weights.
Config files for the various models are presented below. Each of the config files link to a set of weights, all of which can be downloaded from the above link

Model | Config 
:---:|:---:
Trained model before Active Learning | config_cpu_predict_FILE0024.json
Active Learning model 10 percent of data | config_cpu_predict_FILE0024_10p.json 
Active Learning model 15 percent of data | config_cpu_predict_FILE0024_15p.json
Active Learning model 20 percent of data | config_cpu_predict_FILE0024_20p.json
Active Learning model 25 percent of data | config_cpu_predict_FILE0024_25p.json
Active Learning model 30 percent of data | config_cpu_predict_FILE0024_30p.json
Active Learning model 10 percent random data | config_cpu_predict_FILE0024_10p_random.json
Active Learning model 15 percent random data | config_cpu_predict_FILE0024_15p_random.json
Active Learning model 20 percent random data | config_cpu_predict_FILE0024_20p_random.json
Active Learning model 25 percent random data | config_cpu_predict_FILE0024_25p_random.json
Active Learning model 30 percent random data | config_cpu_predict_FILE0024_30p_random.json
Active Learning model trained on all data | config_cpu_predict_FILE0024_TrainAll.json

## Installing

To install the dependencies, run
```bash
pip install -r requirements.txt
```
And for the GPU to work, make sure you've got the drivers installed beforehand (CUDA).

It has been tested to work with Python 2.7.13 and 3.5.3.

## Detection

## Training

### 1. Data preparation 

Get training images

Organize the dataset into 4 folders:

+ train_image_folder <= the folder that contains the train images.

+ train_annot_folder <= the folder that contains the train annotations in VOC format.

+ valid_image_folder <= the folder that contains the validation images.

+ valid_annot_folder <= the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.


### 2. Edit the configuration file
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "min_input_size":       352,
        "max_input_size":       448,
        "anchors":              [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326],
        "labels":               ["raccoon"]
    },

    "train": {
        "train_image_folder":   "/home/andy/data/raccoon_dataset/images/",
        "train_annot_folder":   "/home/andy/data/raccoon_dataset/anns/",      
          
        "train_times":          10,             # the number of time to cycle through the training set, useful for small datasets
        "pretrained_weights":   "",             # specify the path of the pretrained weights, but it's fine to start from scratch
        "batch_size":           16,             # the number of images to read in each batch
        "learning_rate":        1e-4,           # the base learning rate of the default Adam rate scheduler
        "nb_epoch":             50,             # number of epoches
        "warmup_epochs":        3,              # the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically
        "ignore_thresh":        0.5,
        "gpus":                 "0,1",

        "saved_weights_name":   "raccoon.h5",
        "debug":                true            # turn on/off the line that prints current confidence, position, size, class losses and recall
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",

        "valid_times":          1
    }
}

```
### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

### 4. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 5. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

By adding `-W 'T'` you also write all of the bounding box predictions to a file called preds.txt

If you wish to change the object threshold or IOU threshold, you can do it by altering `obj_thresh` and `nms_thresh` variables. By default, they are set to `0.5` and `0.45` respectively.

## Evaluation

`python evaluate.py -c config.json`

Compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.

For evaluting our active learning models we evaluate on the FILE0020 dataset.
