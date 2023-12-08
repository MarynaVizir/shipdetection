# Ship detection
> This CNN-model is being developed to detect ships in satellite images.
> 
> 
## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Shipping traffic is growing fast. More ships increase 
the chances of infractions at sea like environmentally 
devastating ship accidents, piracy, illegal fishing,
drug trafficking, and illegal cargo movement. This has compelled 
many organizations, from environmental protection agencies to 
insurance companies and national government authorities, 
to have a closer watch over the open seas. 
- The **_purpose_** is to build a model that detects all 
ships in satellite images.


## Technologies Used
- Python - version 3.10.5
- numpy - version 1.26.2 
- pandas - version 2.1.3
- tensorflow - version 2.15.0
- seaborn - version 0.13.0  and others


## Features
The ready features here:
- Recognition of the presence of any ship in an real image
- 


## Setup
In order to setup the project install all dependencies listed 
in _**'requirements.txt'**_  file at your local environment.
Use the command:

`$ pip install -r requirements.txt`


## Usage
For model usage you need to download the dataset with images from
https://www.kaggle.com/competitions/airbus-ship-detection/data
to your local directory or you can use appropriate API. 
Please check if the file path in the code _**'model-training.py'**_ 
suits your choice and your OS.
If so, you need to run _**'model-training.py'**_.
Be aware that it may require additional RAM as the dataset is rather 
huge and the image size is 768*768 pixels. Or you can try firstly on 
a smaller random sample.
As a result of _**'model-training.py'**_ execution, the model file 
(HDF5-format) will appear in the root folder. You can use it with 
running _**'model-inference.py'**_ by testing your image instances.


## Project Status
Project is: _in progress_ . 


## Room for Improvement

Room for improvement:
- Perfomance and other CNN techniques

To do:
- Oh.. I discovered (but it was to late) that the goal was to place 
an aligned bounding box segment around the location of the ships... 
So I had to use segmentation and masks. With dice score assessment..
I'm just trying to study how to use this type of architecture yet, so 
unfortunately couldn't build it in time.. Thank you if you read it..((


## Contact
Created by [MarynaVizir](https://github.com/MarynaVizir). 
My [LinkedIn](https://www.linkedin.com/in/maryna-vizir-55402321a).
Feel free to contact me!
