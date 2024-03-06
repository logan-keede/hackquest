# WNS Triange Hackquest Solution

[WNS](https://datahack.analyticsvidhya.com/contest/wns-triange-hackquest/#About)

## Description

This repository contains my solution for the WNS Triange Hackquest, an online analytics hackathon organized by Vidhya Analytics. The solution addresses the provided challenges, showcasing analytical acumen, problem-solving skills, and insights derived from real-life business scenarios.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)


## Installation
Put training and testing images in train and test folder and put the label in corresponding csv files in the same folder.
images in test folder are used for making prediction
run below commands
```bash
conda create --name tf_gpu tensorflow-gpu
pip install pandas
pip install -U efficientnet
```
or just load from [checkpoint](https://drive.google.com/file/d/1k2MBNIy77yM59ENz57qlNx2Qyx4npKrD/view?usp=sharing)

## Usage
run pretrained.py on the dataset
run submit.py if you are using checkpoint


## Features
Use of EfficientB1 pretrained model for prediction of edited insurance claim images.
It is made for WNS but can be used in similar convolution binary classification problem.
test.py contains code for testing.




