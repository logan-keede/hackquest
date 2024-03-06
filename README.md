# WNS Triange Hackquest Solution

[WNS Triange Hackquest](https://datahack.analyticsvidhya.com/contest/wns-triange-hackquest/#About) 

🚀 Achieved Air Rank 81

## Description

This repository contains my solution for the WNS Triange Hackquest, an online analytics hackathon organized by Vidhya Analytics. The solution addresses the provided challenges, showcasing analytical acumen, problem-solving skills, and insights derived from real-life business scenarios.

## Key Insights

- **Data Augmentation:** Implemented advanced data augmentation techniques to enhance model robustness and improve performance.

- **Pretrained Networks:** Leveraged pretrained neural networks to boost model accuracy and efficiency, showcasing the power of transfer learning in analytics scenarios.

- **Loss Functions:** Explored and applied various loss functions tailored to the specific challenges of the provided datasets, optimizing model training for better results.

- **Handling Imbalanced Datasets:** Explored strategies to effectively handle imbalanced datasets, ensuring fair and accurate model predictions in scenarios with disparate class distributions.

## Highlights

- **Data Analysis:** Explored provided datasets.
- **Modeling:** Developed analytical models.
- **Results:** Achieved insights and solutions.



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




