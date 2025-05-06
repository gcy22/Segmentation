Here is the related work of our paper "Enhanced Medical Image Segmentation via Graph - Based Attention and Correlation Graph Convolution" in The Visual Computer.

The following are the steps for using our works:
    We provide the source code for processing the LiTs2017 liver dataset. 
    This is the dataset download address: https://competitions.codalab.org/competitions/17094#participate.


    
    It should be noted that before using train.py, the images in the dataset need to be processed into the form of 1*1*64*256*256. 
        If you want to train your own dataset and the values of D, H, and W do not meet the requirements of your dataset, 
        you can also pre - process the images into the format of 1*128*512*512. 
        You just need to remove the comments from the last layer of the network in Med_Seg_ViG.py in the model folder.



You just need to place your dataset in this folder and arrange it in the following format: 
![image](https://github.com/user-attachments/assets/4e111fd5-c6aa-4916-a455-bfa2ccd41583)


