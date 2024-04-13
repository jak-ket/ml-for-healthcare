# ReadMe for Grad-Cam and its corresponding permutation test
## cnn_grad_cam.ipynb
For Q4, use the cnn_grad_cam.ipynb file. Be sure to download our trained model from the polybox that is linked in the notebook. The ten images we used for visualisation can also be found in the polybox. The link is also at the beginning of the notebook. Make sure that you specify all paths correctly. The pictures on polybox are in a folder called 'img_for_saliency' and are grouped into subfolders based on their classes for the 'ImageFolder' to assign labels correctly.
The Grad-Cam class and all other functionality are defined in the grad_cam.py file, and make sure the functions you are using in the notebook are imported from this file.

# grad_cam_permutation_test.ipynb

For the permutation test for Grad-Cam in Q5, you need the grad_cam_permutation_test.ipynb file. This model builds on our original model, its permutated version, and 10 test images. For all those three components, you can find the link to a polybox at the beginning of the notebook. Ensure the folder structure is correct so the models are correctly imported and the 'ImageFolder' assigns the labels correctly. Again, from the grad_cam.py file, you need the class CNN_grad_cam and the function display_datasets_heatmap to visualise the grad cam saliency maps with the permuted model