# ParkingCarSpaceDetection

### This project deploys the model I trained on my custom dataset to perform better on low-light, rainy and low resolution conditions. Here is the repository where I preprocessed data and trained model: 
### https://github.com/DeepT7/car_detection_in_low-light.git

And here is the repository that I trained the model that enhanced low-light images:
https://github.com/DeepT7/Retinexformer/tree/master

This is a part of smart parking car project.
https://github.com/unt4m1nh/EasyParking

Faster R-CNN model is deployed to detect car in the parking, the output is boxes surrounded cars. From that boxes and spots' coordinates, the program calculates and predicts if the parking spots are empty or occupied.

The precision of the program depends on the deep learning's robustness. So we can improve the precision by training the model with particular datasets.

I have trained the deep learning model on the custom data so that it can perform on low-light conditions much better.
Here is the video of car detection after training model:

https://github.com/DeepT7/ParkingCarSpaceDetection/assets/109886442/339747f4-b2aa-4fdf-a8c8-d718d96067a3


And the final video: 

https://github.com/DeepT7/ParkingCarSpaceDetection/assets/109886442/0f944583-1be1-40a9-bd0e-bc3c10c6c123



The program has reference from the author: 

https://github.com/tempdata73
