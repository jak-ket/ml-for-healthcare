All packages required to run the code are contained in `environment.yml`.  

# Project 1: Interpretable and Explainable Classification for Medical Data
The code for this project is in the 'explainability' folder. The folder structure is as follows:
## Part 1: Heart Disease Prediction Dataset
All code for this part is included in `heart_disease_prediction.ipynb`, the data and column descriptions can be found on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data). The implementation of Neural Additive Models is contained in the folder `/nam` and taken from Google Research [github.com/lemeln/nam](https://github.com/lemeln/nam).

## Part 2: Pneumonia Prediction Dataset

### Q1-Q3: Data Analysis, CNN Classifier and Integrated Gradients
All code for this part can be found in `cnn_imaging.ipynb`.

### Q4: Grad-CAM
For Q4, use the `cnn_grad_cam.ipynb` file. Be sure to download our trained model from the polybox that is linked in the notebook. The ten images we used for visualisation can also be found in a polybox. The link is also at the beginning of the notebook. Make sure that you specify all paths correctly. The pictures on polybox are in a folder called 'img_for_saliency' and are grouped into subfolders based on their classes for the 'ImageFolder' to assign labels correctly.
The "CNN_grad_cam" class and all other functionality are defined in the `grad_cam.py` file, and make sure the functions you are using in the notebook are imported from this file.

### Q5: Data Randomization Test
For the permutation test for Grad-Cam in Q5, you need the `grad_cam_permutation_test.ipynb` file. This notebook builds on the permutated model and 10 test images. For all those two components, you can find the link to a polybox at the beginning of the notebook. Ensure the folder structure is correct so the models are correctly imported and the 'ImageFolder' assigns the labels correctly. Again, from the `grad_cam.py` file, you need the class "CNN_grad_cam" and the function display_datasets_heatmap to visualise the Grad-Cam saliency maps with the permuted model. In order the perform the data randomization test on Integrated Gradients, refer to `cnn_permuted.ipynb`. This mostly follows the structure of the Grad-Cam notebook using techniques and packages introduced in 2.3.

# Project 2: Time Series and Representation Learning

## Part 1: Supervised Learning on Time Series 
All code for this part is included in `time_series" folder. We use the following datasets:
ThePTBDiagnostic 

● MIT-BIH Arrhythmia Database (https://physionet.org/physiobank/database/mitdb/)

● ECG Database (https://physionet.org/content/ptbdb/1.0.0/)

The folder structure is as follows:

### Q1: Exploratory Data Analysis, Q2: Classical machine learning methods
All code for this part can be found in `eda1.ipynb` and `classic_ml1.ipynb`.


### Q3: Recurrent Neural Networks
Code to train RNN with LSTM can be found in `rnn.ipynb` and `rnn.py` contains model class and training function. Code for the bidirectional LSTM can be found under `ts_bilstm.ipynb`. The model is under `models/bilstm.pth`.

### Q4: Convolutional Neural Network
Code to train the two CNN models (vanilla and with residual blocks) can be found at `ts_cnn.ipynb`. The corresponding models are stored at `models/vanillacnn.pth` and `models/residualcnn.pth`.

### Q5: Attention and Transformers
Code to train the Transformer and visualize the attention maps can be found in `transformer.ipynb`. The model classes and training function can be found in `transformer.py`.

## Part 2: Transfer and Representation Learning

### Q1: Supervised Model for Transfer
Transfer learning for the CNN encoder is covered in the first half of the notebook `umap.ipynb`. We create the model `transfercnn.pth`. The embeddings are stored in the polybox under https://polybox.ethz.ch/index.php/s/bu0w5P5x9DHU86G

###  Q2: Representation Learning Model 
Our autoencoder model is defined in `models/encoder.py`. The training and evaluation is done in `representation_learning.ipynb`.


### Q3: Visualising Learned Representations 
Code for the UMAP representations for the CNN encoder can be found under `umap.ipynb`. 
Our code for the kulback-leibler divergence is found in `models/kl.py`
The representations of the autoencoder are visualised in `viualize_encoder.ipynb`

### Q4: Fine-tuning Strategies
The notebook `fine_tuning_encoder_q1.ipynb` contains code for classic ML and MLP strategies using encoder from Q1 and `fine_tuning_encoder_q2.ipynb` using encoder from Q2 (Autoencoder). 
