# Introduction

Being able to extract certain background information about the users, such as their age, can be useful in numerous ways including user-profiling and personalised call-routing. With this knowledge, one possible application of speech recognition age classifiers is a recommendation system that can return an age-relevant result when asked for a movie recommendation. This enhances the quality of recommendation since a popular movie for the general audience could be an irrelevant movie to a child requesting for a children’s show with a similar title

## Problem Statement - Age Classification from Voice Samples

In this project we have thus experiemented with multiple neural network architectures to train an age classifier from voice recording samples. They range from vanilla Recurrent Neural Networks (RNNs), Bidirectional Gated Recurrent Units (Bidirectional GRUs), Temporal Convolutional Neural Networks (TCNs), and vanilla Convolutional Neural Networks (CNNs). We have found the most success using vanilla CNNs with an accuracy of 35%.

## Dataset used

The dataset to be used is found in [Common Voice](https://commonvoice.mozilla.org/en/datasets), a database for speech recognition software by Mozilla. The specific version used is Common Voice Corpus 8.0, with the audio language set to English. This particular dataset is 70 GB large and consists of 79,398 Voices (with about 20GB of 450,000 files after cleaning and removing 75% of the raw input files) in MP3 format, with the breakdown of each age group as follows before data cleaning:
| Age Group | < 19 | 20-29 | 30-39 | 40-49 | 50-59 | 60 -69 | 70-79 | 80 -89| >90|
| ---------------------- | ---- | ----- | ----- | ----- | ----- | ---- |---- |---- |---- |
| Percentage in Dataset | 6% | 24% | 13% | 6%| 9%| 4%| 1%| ~0%| ~0%|

The breakdown of each age group after data cleaning is as follows:
| Age Group | < 19 | 20-29 | 30-39 | 40-49 | 50-59 | > 60|
| ---------------------- | ---- | ----- | ----- | ----- | ----- | ----|
| Percentage in Dataset | 16.7% | 16.7% |16.7% |16.7%| 16.7%| 16.7%|

## Final Model - Convolutional Neural Network (CNN)

Among the models we have experimented with, the model that produced the highest accuracy (35%) was a CNN with a feature learning block (FLB) consisting of convolution, pooling, and batch normalisation layers. Convolution layers extract local and global high level features whereas pooling layers decrease processing time by reducing the dimensionality of feature maps. We wanted to arrange the layers in a way that builds robust and salient feature maps.

For the first convolutional layer, a **kernel size of 3x3 and stride 2** was used. ReLU was then used as the activation function in this layer as well as all other convolutional layers to generalise the model to achieve better performance during training. **Batch normalisation** was then performed, followed by **max pooling with a kernel size and stride of 2x2**. For the second convolutional layer, **a kernel size of 5x5 with stride 1x1** was used. **Max pooling was performed again with a kernel size of 2x2 and stride 1x1**. Finally, for the third convolutional layer, a **kernel size of 2x2 and stride 1x1** was used, and another round of batch normalisation was conducted before finally parsing its output into a fully connected layer for the final output.

### Results

Here we can see that both the training and validation accuracy achieving 35% accuracy
![CNN Results ](images/CNN_Results.png)

# How to Run the Code

Connect your runtime to GPU. Avoid running the training for too many epochs as you might hit the Colab usage limits.

## Downloading the Audio Dataset

1. Open ProjectCode_Riley from https://drive.google.com/drive/folders/196KRvyUlmY05-hxnRYhLqB8B34eCLUel?usp=sharing.
2. Add shortcut link from this folder to a folder in your local Drive
3. Add the path of your local Drive folder to the code and update line 15
   <img width="1202" alt="Screenshot 2022-04-24 at 11 58 47 PM" src="https://user-images.githubusercontent.com/62118373/164985161-861b6531-4e80-4130-b9de-88f788f76bb6.png">
4. Run the next block of code  
   <img width="787" alt="Screenshot 2022-04-25 at 12 02 19 AM" src="https://user-images.githubusercontent.com/62118373/164985291-8511acf1-0d0d-41d0-99d6-479cd796ccf0.png">
5. Unzip and remove the zip file
   <img width="413" alt="Screenshot 2022-04-25 at 12 01 28 AM" src="https://user-images.githubusercontent.com/62118373/164985250-600ec8ec-0c9d-475e-a44f-504b30505cf8.png">

## Running RNN and Bidirectional GRU Code

Run the code all the way till this block.
<img width="1185" alt="Screenshot 2022-04-25 at 12 09 37 AM" src="https://user-images.githubusercontent.com/62118373/164985559-cf014e1e-fb7d-4b61-b2f1-f4189133037c.png">
If you want to run the RNN model, you have to change `line 8` from `BidrectionalGRU()` to `RNNModel()` Otherwise, you want to run the Bidirectional GRU model, you don't have to change anything

## Training

To train the model, run this block of code. For every epoch, a validation will be performed on the model. You can refresh the tensorboard to see the validation results and find out how your model is performing. Avoid running the training for too many epochs as you might hit the Colab usage limits.
<img width="943" alt="Screenshot 2022-04-25 at 12 17 20 AM" src="https://user-images.githubusercontent.com/62118373/164985894-3dc184b0-6526-4285-939f-568476177314.png">

## Saving your Model

Run this block of code to save your model as a `.pth` file
<img width="810" alt="Screenshot 2022-04-25 at 12 19 38 AM" src="https://user-images.githubusercontent.com/62118373/164986005-dce39a6d-3418-4eb8-8e40-f978849046fd.png">

## Testing

After training, run your model against the test set. You will be able to see the accuracy and loss of your model once it's done.
<img width="600" alt="Screenshot 2022-04-25 at 12 20 17 AM" src="https://user-images.githubusercontent.com/62118373/164986030-aebc96a3-d2b2-4e64-aac8-597defce8a65.png">
