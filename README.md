
## Introduction
According to the World Health Organization (WHO), around 6.1% of the world's population, or 466 million people, have hearing loss. Of these individuals, 34 million are children and most live in low- and middle-income countries. About one-third of people over the age of 65 have disabling hearing loss. It is estimated that by 2050, over 900 million people worldwide will have disabling hearing loss. There are 300 different sign languages in the world.  

To tackle the issue of communication gap between the hearing and the deaf community, sign language interpreters should be in use. However, it is difficult to have everyone learn different sign languages. But what if there was a software application that could interpret different sign languages and help everyone understand each other?  

Keeping this idea in mind, this project focuses on bridging this gap of communication between the deaf and mute, and hearing communities.  

For the sake of simplicity of this project, we will be using American Sign Language (ASL) as the targeted language.  

---

## What is American Sign Language (ASL)?
Sign languages, like spoken languages, vary across the world and are unique to specific regions or cultures. These visual languages evolve naturally and are influenced by the spoken languages of their respective communities. American Sign Language (ASL) serves as the primary language for many members of the deaf and hard-of-hearing communities in the United States and parts of Canada.  

---

## What is an ASL Interpreter?
An ASL Interpreter is a CNN model that interprets signs made by users of ASL. It is trained using a dataset containing various hand gestures corresponding to the English alphabet.  

The dataset undergoes preprocessing and is utilized for training, validation, and testing phases. Once trained, the model can predict the corresponding signs in real time using input from a live camera feed.  

Python libraries used:  
- numpy  
- pandas  
- mediapipe  
- openCV  
- sklearn  
- torch  

---

## Hypothesis
A CNN-based ASL detector, trained on a well-structured and comprehensive dataset of ASL gestures, can accurately interpret and classify hand gestures in real time, providing a reliable and efficient tool for communication between ASL users and non-signers.  

---

## Environment Setup
- IDE: Visual Studio Code (VSCode)  
- Dataset: [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
- Dataset contains:  
  - 26 alphabet directories (A–Z)  
  - 3 additional classes: space, delete, nothing  
  - ~87,000 images total (3,000 per class)  

Libraries:  
`numpy, pandas, mediapipe, sklearn, torch`  

---

## Methodology
The whole process of building the model is divided into four steps:  
1. **Data Preprocessing**  
2. **Model Building**  
3. **Model Training and Evaluation**  
4. **Real-time Prediction**  

### A. Data Preprocessing
- Images normalized and resized to `64x64`.  
- One-hot encoding applied to labels.  
- Dataset split into 70:15:15 (train : validation : test).  

### B. Model Building
- CNN Model with the following layers:  
  - Conv2D (32 filters, 3x3) → ReLU → MaxPooling (2x2)  
  - Conv2D (64 filters, 3x3) → ReLU → MaxPooling (2x2)  
  - Conv2D (128 filters, 3x3) → ReLU  
  - Flatten → Fully Connected (128 neurons)  
  - Fully Connected → Output classes  

![model-architecture](https://github.com/user-attachments/assets/67319aa4-4fda-4090-853f-c601b731e075)


### C. Model Training
- Loss: Cross Entropy Loss  
- Optimizer: Adam  
- Metric: Accuracy  
- Training: 5 epochs  

### D. Real-Time Prediction
- Used **MediaPipe** to extract hand landmarks.  
- Landmarks fed into trained CNN model.  
- Model predicts the corresponding sign in real time.  


![image1](https://github.com/user-attachments/assets/1eaf4ea8-a950-4600-be43-5ad825a34ae1)

![image2](https://github.com/user-attachments/assets/95d79c6d-f2c6-4878-9803-50eecb65dc10)

![image3](https://github.com/user-attachments/assets/e76ef32b-6453-4013-9a84-8f696323fcb7)


## Evaluation Metrics Comparison
Our model vs a reference research paper:  

| Evaluation Metrics | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| **Model A (ours)** | 99.42%   | 99.39%    | 99.38% | 99.38%   |
| **Model B (paper)**| 98.6%    | 99%       | 99%    | 99%      |

Reference Paper: [Real-Time Sign Language Detection using CNN](https://www.researchgate.net/publication/364185120_Real-Time_Sign_Language_Detection_Using_CNN)  

---

## Results
### A. Confusion Matrix

![confusion-matrix](https://github.com/user-attachments/assets/68a3c9b4-b1a4-4f6f-8024-234753ba694d)


### B. Accuracy and Loss Curve
![accuracy-loss](https://github.com/user-attachments/assets/42af30d4-47a9-4a05-9342-fbb5ef88f183)


Final test set results:  

| Evaluation Metrics | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| **Model A (ours)** | 99.42%   | 99.39%    | 99.38% | 99.38%   |

---

## Conclusion
Understanding sign language is crucial for communication and necessary to bridge the gap between the deaf, mute, and hearing communities.  

This project successfully demonstrates the ability of a CNN-based ASL Interpreter to classify hand gestures with high accuracy in real time.  

Through this, I hope to continue exploring Machine Learning, Deep Learning, and Computer Vision for even more impactful applications.  
