from scipy.stats import spearmanr
import numpy as np  

def _calculate_similarity_of_heatmaps(heatmap1, heatmap2):
    rho, _ = spearmanr(heatmap1.flatten(), heatmap2.flatten())
    return rho

def similarity_of_images(model, model_permuted, dataloader, heatmap_fun) -> list:
    similarities = []
    model.cpu()
    model_permuted.cpu()
    for img, _ in dataloader:
        heatmap1 = heatmap_fun(model, img)
        heatmap2 = heatmap_fun(model_permuted, img)
        similarities.append(_calculate_similarity_of_heatmaps(heatmap1, heatmap2))
    return similarities 