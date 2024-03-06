# WNS Triange Hackquest Solution

[WNS](https://datahack.analyticsvidhya.com/contest/wns-triange-hackquest/#About)

## Description

This repository contains my solution for the WNS Triange Hackquest, an online analytics hackathon organized by Vidhya Analytics. The solution addresses the provided challenges, showcasing analytical acumen, problem-solving skills, and insights derived from real-life business scenarios.

## Table of Contents

- [Installation]
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
or just load from checkpoint in below link

## Usage
run pretrained.py on the dataset
run submit.py if you are using checkpoint
I tried to upload the model for diret usage but github wont allow it so here is the drive link:- 

## Features
Use of EfficientB1 pretrained model for prediction of edited insurance claim images.
It is made for WNS but can be used in similar convolution binary classification problem




