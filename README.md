# Interpretable and Explainable Classification for Medical Data
The tasks we worked on are described in `tasks.pdf`, our answers in `report.pdf`. All packages required to run the code are contained in `environment.yml`. 

## Part 1: Heart Disease Prediction Dataset
All code for this part is included in `heart_disease_prediction.ipynb`, the data was provided via Moodle and the columns are described on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data). The implementation of Neural Additive Models is contained in the folder `/nam` and taken from Google Research [github.com/lemeln/nam](https://github.com/lemeln/nam).

## Part 2: Pneumonia Prediction Dataset

### Q1-Q3: Data Analysis, CNN Classifier and Integrated Gradients
All code for this part can be found in `cnn_imaging.ipynb`.

### Q4: Grad-CAM
For Q4, use the `cnn_grad_cam.ipynb` file. Be sure to download our trained model from the polybox that is linked in the notebook. The ten images we used for visualisation can also be found in a polybox. The link is also at the beginning of the notebook. Make sure that you specify all paths correctly. The pictures on polybox are in a folder called 'img_for_saliency' and are grouped into subfolders based on their classes for the 'ImageFolder' to assign labels correctly.
The "CNN_grad_cam" class and all other functionality are defined in the `grad_cam.py` file, and make sure the functions you are using in the notebook are imported from this file.

### Q5: Data Randomization Test
For the permutation test for Grad-Cam in Q5, you need the `grad_cam_permutation_test.ipynb` file. This notebook builds on the permutated model and 10 test images. For all those two components, you can find the link to a polybox at the beginning of the notebook. Ensure the folder structure is correct so the models are correctly imported and the 'ImageFolder' assigns the labels correctly. Again, from the `grad_cam.py` file, you need the class "CNN_grad_cam" and the function display_datasets_heatmap to visualise the Grad-Cam saliency maps with the permuted model. In order the perform the data randomization test on Integrated Gradients, refer to `cnn_permuted.ipynb`. This mostly follows the structure of the Grad-Cam notebook using techniques and packages introduced in 2.3.
