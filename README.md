# **Cloudphysician's The Vital Extraction Challenge**

## **Problem Statement:**
Patient monitoring is crucial in healthcare, as it allows healthcare professionals to closely track a patient's vital signs and detect any potential issues before they become serious. In particular, monitoring a patient's vitals, such as heart rate, blood pressure, and oxygen levels, can provide valuable information about a patient's overall health and well-being. The core problem statement is to extract *Heart Rate, SpO2, RR, Systolic Blood Pressure, Diabolic Blood Pressure, and MAP* from images of ECG Machines.

## **Proposed Pipeline:**
The proposed pipeline consists of a total of four stages. The whole pipeline is made such that it can work in two modes: **'Fast'** and **'Accurate**'. 
**By default, the pipeline is set to 'Accurate' mode**. 

The 'Fast' mode ensures that the whole pipeline runs under 2 seconds on CPU, while the 'Accurate’ mode ensures maximum accuracy at each stage, thus ensuring the best overall performance. 


 The four stages of the pipeline are as follows:

![Pipeline](https://github.com/Umang1815/CloudPhysician-s-Vital-Extraction-Challenge/blob/main/pipeline.PNG)



1. Screen Extraction
2. Number Detection
3. OCR
4. Number Classification



# Screen Extraction


The first stage of the pipeline is the extraction of monitor screens from the given images. We have approached the problem in two ways, which are as follows:
-  #### **Corner Regression**: 

    The Corner Regression approach uses a CNN encoder to regress on the values of the bounding boxes of the desired screen, thereby producing an output vector of length eight (x1,y1,x2,y2,x3,y3,x4,y4 - coordinates of the bounding box). The followings are the details of the training parameters and augmentations for the CNN:

   *  Model: **Fine-Tuned Resnet 34**
 * Augmentations: 

    We used a few custom augmentations in our pipeline because the standard libraries (e.g., Albumentations) do not provide augmentations on the random 4-sided polygon and are specifically tuned to rectangles. These custom augmentations improved the performance of our model and made it more robust.

    **Custom Horizontal Flip**: Our custom augmentation performs a horizontal flip on the image and accordingly changes the bounding box coordinates, even in non-rectangular cases for the newly flipped image.

    **Custom Random Crop**: Our custom implementation performs random crops to ensure that the full area of the screen is always visible in the augmented image. This random crop is implemented to mimic the scenarios where the model generally performs poorly (i.e., the original model performed poorly in images where the monitor was too close to the edge of the image). This augmentation significantly improved the model's performance on the edge cases, making it more robust to the actual scenarios.

    **RandomBrightnessContrast**: RandomBrightnessContrast from the Albumentations library for more robust training on various input image scenarios. 

**Loss: Mean Squared Error**

**Learning Rate: 0.001**

**Epochs: 30**

**Mean IoU: 0.873** 

* #### **UNet++ Semantic Segmentation**:

UNet++ takes the original image as input and mask as ground truth. We generated the ground truth mask from the given bounding box coordinates using the cv2.fillPoly function. The details of the training, preprocessing, and post-processing are as follows:

* Model: **UNET++ (from segmentation_models_pytorch)**
* Encoder: **resnext101_32x8d**
* Encoder weights: imagenet
* Augmentations:
    Except for the augmentations used in "Corner Regression," we used one more augmentation for training purposes: ShiftScaleRotate: translate, scale, and rotate the input. 
* Preprocessing:
    We use the same preprocessing as done on the original UNET++ pretraining. This normalizes the image to a distribution accustomed to the pre-trained weights.

**Epochs: 15**

**Learning Rate: 0.001**

**Mean IoU: 0.896**

**Loss: 0.75*Dice Loss + 0.25*BCE Loss**


We made a custom weighted loss for the training of UNET++. Dice Loss increases the intersection with the ground truth image, while BCE Loss ensures confidence in predictions. This weighted average of losses resulted in better overall learning of the model.






* #### **Area Union Fusion Ensembling**:
    We used the predicted boxes from Corner Regression (I) and UNET++ (II) to ensemble for final prediction. We used Area Union Prediction method, i.e, the largest polygon that covers both the predicted polygons, thus ensuring maximum coverage.

    ***Mean IoU Score: 0.91***



# Number Detection + OCR
After extracting the screens, our next stage in the pipeline is to find all the numbers on the screen and return the appropriate bounding boxes across them. This task is achieved with the help of two separate models: 
  
  - **YOLOv5** 
  - **PaddleOCR**
      
      * YOLOv5: 
              
          We trained YOLOv5 model for the detection of numbers on a screen. We assigned a “number” class to all given bounding box and then fine-tuned YOLO for the same. The details of training and hyperparameters are as follows:

          * Model: YOLOv5
          * Size: M
          * Epochs: 25
          * Learning Rate: 0.01 
          * Mean Inference time: 650 ms
          * Average Precision: 0.77

      * PaddleOCR:
              
         PaddleOCR is an optical character detection and recognition model implemented using PaddlePaddle (PArallel Distributed Deep LEarning) framework. In our pipeline, we have used Pre-trained PaddleOCR to detect numbers from the extracted screen. The details of the hyperparameters are as follows:
          * Recognition Algorithm: CRNN
          * Detection Algorithm: DB
          * Mean Inference Time: 2.5 s

    * Weighted Box Fusion of PaddleOCR and YOLO:
              
         Fine-tuned YOLOv5 is showing promising results on layouts it was trained on but a relatively low accuracy on unseen layouts. In any case, it was not giving any noise in the prediction. Pre-trained PaddleOCR captures boxes of all numbers on the screens, but being a text recognition model, it also predicts few noise in the screens, which is not required. Hence, we ensemble the predictions of both the models, using Weighted Box Fusion algorithm, taking the good points of both algorithms, thereby resulting in much more robust predictions.
      * Weighted Box Fusion algorithm utilizes confidence scores of all proposed bounding boxes to construct the averaged boxes. 

       * Mean Inference Time - 3.3 s




![WBF](https://github.com/Umang1815/CloudPhysician-s-Vital-Extraction-Challenge/blob/main/download.png)

After getting all the bounding boxes around the numbers in the image, we use Optical Character Recognition to get the numbers written in the images. We implement this task using the ParSeq (Permuted AutoRegressive SEQuence) model. We are using the pre-trained model of ParSeq, and the details are as follows:

  * **Model: ParSeq**
  * **Accuracy: 0.95**
  * **Mean Inference Time: 1 s**

# Number Classification

This task consists of classifying all the numbers detected by OCR into their respective classes. To implement this task, we have made **CRABBNet (Custom Recognition Assisted Bounding Box classification Network)**.  
  CRABBNet takes 3 inputs, a 4-channel Image, all numbers on the screen, and the target number which needs to be classified. 

  The working of CRABBNet is explained below:
  
  * The screen-extracted image (of 3 RGB channels) on which the prediction has to be made is concatenated with a 1-channel mask of the bounding box of the target number to produce a 4-channel input for the network.

  * This 4-channel input is passed to a ResNeXt - 50 model, which produces a 2048-length feature vector of the image. 
A 14-length vector is created, which contains information about all the numbers on the screen.

  * The 10 numbers in the vector denote the 10 numbers which are present on the screen. If less numbers are present, the remaining values are filled with 0
The next 3 numbers are binary values, indicating presence of ‘/’, ‘(‘ and ‘)’  in our target number which is to be classified. This is because, many a times in prediction of DBP and SBP, ‘/’ is seen in the image, while in prediction of MAP, ‘(‘ and ‘)’ are found.

* The last number of the vector is the target number itself.

* This 14-length vector is then passed to a linear layer, which converts this into 6-length vector. 
* This 6-length vector, is concatenated with the previous 2048-length feature vector, thereby resulting in 2054-length vector.
* This 2054-length vector is again concatenated with the target number, to give more weight to the target number, informing model to focus on predicting the target number.
* Hence, after this step, we would result in 2055-length vector.
This 2055-length vector is then passed again to a linear layer, with softmax activation, which gives probability of 6 classes among which the number would be present.

    * **Training Methodology**:
        * **Model : Resnext50_32x4d**
        * **Epochs : 15**
        * **Learning_rate : 0.001**
        * **Loss : CrossEntropy** 
        * **Accuracy :    77% (Out-of-fold Layout Validation) and 98% (Single Layout training)**





#### **Custom Logit-Decoder for Multi-label Matching**:

Using the above model we generate 6-class predictions for each bounding box on the screen. To combine predictions from multiple bounding boxes in multiple classes while ensuring single prediction for each vital, we use a customized probability decoder.


We use variance measure across each class prediction to get the most confident class and decide the box prediction for that class. This step is repeated on each class in the order of decreasing variance while simultaneously already selected bounding boxes to ensure a one-to-one mapping of bounding boxes and classes.


