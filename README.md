# CS 189 Project T

This is our submission for [CS 189 Project T](https://www.notion.so/CS189-289A-Final-Project-f6da6223f8ce4722817ad6e272cc45d5)

Project Members: Charlie Snell, Arnav Gudibande, Divi Schmidt

### Installation

Install and activate [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
* ```python3 -m pip install --user virtualenv```
* ```python3 -m venv env```
* ```source env/bin/activate```

Install required dependencies and files
* ```pip install -r requirements.txt```
* ```python project/finetune/cifar10_download.py```

Run Jupyter Notebook
* ```jupyter notebook project```

### Navigating Repository
* ```project/problem1-3.ipynb``` - 3 Jupyter notebooks intended to be completed by students
* ```project/finetune/cifar10_models``` - Source code for CIFAR-10 pre-trained models (source: https://github.com/huyvnphan/PyTorch_CIFAR10)
* ```docs/``` - Contains lecture slides, notes, and quiz

### Learning Objectives
* ```problem1.ipynb``` - In this notebook, students will discover just how powerful the learned features from pretrained models really are. 
They will be comparing the performance of 3 different sets of features. The first is the most naive feature set 
possible: the set of raw pixel values. Then, they will augment the set raw pixel values 
with a set of features extracted from custom convolutional filters. Lastly, they will use different techniques to compare 
these features to the features extracted from a state of the art pretrained model. By taking this "feature" 
based approach, students will be able to leverage concepts from EECS 16AB and earlier parts of 16ML to develop
an understanding of the motivation for pre-trained models

* ```problem2.ipynb``` - This notebook provides a practical exercise of one of the most common use-cases of pre-trained
models -- finetuning. Students will use PyTorch to setup a machine learning workflow and finetune a pre-trained model 
on the simple task of binary image classification. Then, they will look at a specific case of when pre-trained models really shine
-- when there is a lack of available training data. This assignment will not only help build practical skills 
associated with ML development, but will also build intuition for the situations in which pre-trained models are superior
to other alternatives

* ```problem3.ipynb``` - This notebook expands on the use-cases for pre-trained models and highlights a variety of situations
in which pre-trained models can be used -- fine-tuning on dissimilar datasets, fine-tuning different combinations of layers, etc.
The main takeaway for students from this notebook is that there are many use cases for pre-trained models, but there is 
not a "one-size fits all" solution. Students will also come away with a better understanding for the difference between 
features in earlier/later layers and other pre-trained models that they can use


