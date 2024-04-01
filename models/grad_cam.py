import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np 

class CNN_grad_cam(nn.Module):
    def __init__(self, model_trained):
        super(CNN_grad_cam, self).__init__()
        # mount model to cpu for later numpy
        self.cnn = model_trained.to('cpu')
        #get all conv layers excluding max pooling
        self.features = self.cnn.features[:30]
        self.max_pool = self.cnn.features[30]
        self.avg_pool = self.cnn.avgpool
        self.classifier = self.cnn.classifier
        # gradients for hook
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
    def forward(self, x):
        x = self.features(x)
        x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self, x):
        return self.features(x)


def heatmap_grad_cam(grad_cam : CNN_grad_cam, img) -> torch.Tensor:
    # get the most likely class
    grad_cam.eval()
    # allow gradients to be computed
    for param in grad_cam.features.parameters():
        param.requires_grad = True
    # get the gradiants
    gradiants = _get_gradiants(grad_cam, img)
    # pool gradients
    pooled_gradients = torch.mean(gradiants, dim=[0, 2, 3])
    #get activations
    activations = _get_activations(grad_cam, img, pooled_gradients)
    #construct heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on heatmap
    #heatmap = np.maximum(heatmap, 0) # removed for now ReLu function because we only have two classes for one class no characterisics are in the image
    #catch div by zero
    if torch.max(heatmap) != 0:
        heatmap /= torch.abs(torch.max(heatmap))
    return heatmap

def heatmap_in_image(img, heatmap, threshold, mixture, mn, mx) -> np.array:
 
    mask = heatmap > threshold
    # Apply the mask to the tensor
    heatmap_cliped = heatmap.clone()
    heatmap_cliped[~mask] = 0
    heatmap_cliped = heatmap_cliped.numpy()
    img = img.squeeze().numpy()
    img = np.transpose((img - mn) / (mx - mn), (1, 2, 0))   
    heatmap_plotting = cv2.resize(heatmap_cliped, (img.shape[1], img.shape[0]))
    heatmap_plotting = np.uint8(255 * heatmap_plotting)
    # Apply a colormap to the heatmap
    heatmap_plotting = cv2.applyColorMap(heatmap_plotting, cv2.COLORMAP_HSV)/255
    img_clipped = np.clip(img, 0, 1)
    image_weight = mixture
    # Blend the heatmap with the original image
    superimposed_img = (1 - image_weight) * heatmap_plotting + image_weight * img_clipped
    return superimposed_img

def _get_gradiants(grad_cam : CNN_grad_cam, img):
    pred = grad_cam(img)
    #calculate gradients for most likely class 
    pred[:,pred.argmax(dim=1)].backward()
    gradiants = grad_cam.get_activations_gradient()
    return gradiants


def _get_activations(grad_cam : CNN_grad_cam, img, pooled_gradients):
    activations = grad_cam.get_activations(img).detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    return activations


def display_datasets_heatmap(test_grad_cam, dataloader,threshold=0.6, mixture=0.9, classes=('NORMAL','PNEUMONIA')):
    # calculate the number of rows and columns for subplots
    n = len(dataloader.dataset)
    num_rows = 2
    num_cols = n // 2
    
    # create the subplots with reduced vertical spacing
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 7), gridspec_kw={'wspace': .5, 'hspace': 0.5})
    mn = min([dataloader.dataset[i][0].min() for i in range(n)]).numpy()
    mx = max([dataloader.dataset[i][0].max() for i in range(n)]).numpy()
    for i, (img, label) in enumerate(dataloader):
        if i == n:
            break
        heatmap = heatmap_grad_cam(test_grad_cam, img)
        joint_img = heatmap_in_image(img=img, heatmap=heatmap, threshold=threshold, mixture=mixture, mn=mn, mx=mx)
        #print the prediction
        test_grad_cam.eval()
        print(test_grad_cam(img).argmax(dim=1).item(), label.item())
        # calculate the row and column indices for the current subplot
        row_idx = i // num_cols
        col_idx = i % num_cols
        
        # plot the image in the corresponding subplot
        axs[row_idx, col_idx].imshow(joint_img) 
        axs[row_idx, col_idx].set_title(classes[label])
        axs[row_idx, col_idx].axis('off')
        
    plt.show()
# Blend the heatmap with the original image

