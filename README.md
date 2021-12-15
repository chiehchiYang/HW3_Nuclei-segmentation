# hw3_VRDL_Readme


Nuclei segmentation

> This is the hw3 of Selected Topics in Visual Recognition using Deep Learning (2021).
> This homework, I use Mask RCNN to train the model.
## Installation

You can use conda to create a new virtual enrivonment.
And then install the [Pytorch](https://pytorch.org/) from the official webside.


```shell
$ conda create -n hw3 python=3.7
$ conda activate hw3
$ git clone 
$ cd HW3_Nuclei-segmentation
```
Then install the [pycococreator](https://github.com/waspinator/pycococreator)
```shell
$ sudo apt-get install python3-dev
$ pip install cython
$ pip install 
$ git+git://github.com/waspinator/coco.git@2.1.0
```
and opencv
```shell 
$ pip install opencv-python
```
And then install detectron2 from the [official link](https://github.com/facebookresearch/detectron2)


## Data download and Prepare for training 
Please download the dataset from this [google drive  link](https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view?usp=sharing) and put the file below to the pycococreator/examples/train
![](https://i.imgur.com/f9D06D4.png)

And then run 
```sehll
cd pycococreator/examples/shapes
$ python tococo.py
```
You will get the json file name "instances_train2017.json" and please put this annotation to the folder datasets/coco/annotations

![](https://i.imgur.com/9w6JDW8.png)


## Train the model
Now we are ready to train the model.

```shell
$ python train.py 
```
It will generate the results to detectron2_results

## detect test images and conver the output files to coco formate



We can run test.py to produce the results.

```shell
$ python test.py
```

To reproduce my result.
Please download the data from this [google drive  link](https://drive.google.com/file/d/1GiAX6YMZGk2z8WhqtBpLdq3-FHwrUSh8/view?usp=sharing) and put the file below to the detectron2_results

And then run test.py

The results can be found in the folder detectron2_results



## Reference 
- [pycococreator](https://github.com/waspinator/pycococreator)
- [detectron2](https://github.com/facebookresearch/detectron2)