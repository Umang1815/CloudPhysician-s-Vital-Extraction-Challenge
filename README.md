# **Cloudphysician's The Vital Extraction Challenge**

## **Problem Statement:**
Patient monitoring is crucial in healthcare, as it allows healthcare professionals to closely track a patient's vital signs and detect any potential issues before they become serious. In particular, monitoring a patient's vitals, such as heart rate, blood pressure, and oxygen levels, can provide valuable information about a patient's overall health and well-being. The core problem statement is to extract *Heart Rate, SpO2, RR, Systolic Blood Pressure, Diabolic Blood Pressure, and MAP* from images of ECG Machines.

## **Proposed Pipeline:**
The proposed pipeline consists of a total of four stages. The whole pipeline is made such that it can work in two modes: **'Fast'** and **'Accurate**'. 
**By default, the pipeline is set to 'Accurate' mode**. 

The 'Fast' mode ensures that the whole pipeline runs under 2 seconds on CPU, while the 'Accurate’ mode ensures maximum accuracy at each stage, thus ensuring the best overall performance. 


 The four stages of the pipeline are as follows:

![Pipeline](https://github.com/Umang1815/CloudPhysician-s-Vital-Extraction-Challenge/blob/main/pipeline.png)



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




![image.png](data:image/png;base64,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)


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


