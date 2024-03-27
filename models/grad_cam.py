import cv2
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
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    return heatmap

def heatmap_in_image(img, heatmap, threshold, mixture) -> np.array:
    #only keep values of the 14*14 array with value greater than 0.5
    mask = heatmap > threshold
    # Apply the mask to the tensor
    heatmap_cliped = heatmap.clone()
    heatmap_cliped[~mask] = 0
    heatmap_cliped = heatmap_cliped.numpy()
    img = img.squeeze().numpy()
    img = np.transpose(img, (1, 2, 0))
    heatmap_plotting = cv2.resize(heatmap_cliped, (img.shape[1], img.shape[0]))
    heatmap_plotting = np.uint8(255 * heatmap_plotting)
    # Apply a colormap to the heatmap
    heatmap_plotting = cv2.applyColorMap(heatmap_plotting, cv2.COLORMAP_HSV)/255
    img_clipped = np.clip(img, 0, 1)
    image_weight = mixture
    # Blend the heatmap with the original image
    superimposed_img = (1 - image_weight) * heatmap_plotting + image_weight * img_clipped
    superimposed_img= superimposed_img / np.max(superimposed_img)
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
