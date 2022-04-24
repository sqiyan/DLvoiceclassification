# DLvoiceclassification
Code for our Deep Learning project on voice classification by age

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
If you want to run the RNN model, you have to change line 8 from `BidrectionalGRU()` to `RNNModel()` Otherwise, you want to run the Bidirectional GRU model, you don't have to change anything


