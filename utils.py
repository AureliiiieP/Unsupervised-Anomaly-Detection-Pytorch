import cv2
import numpy as np
from torchvision import transforms

def save_inference(nw_output, output_path):
    """Converts and saves model output to specified path.
    """
    img = transforms.ToPILImage()(nw_output.squeeze_(0))
    img.save(output_path)

def save_anomaly_overlay(reference_img, reconstruction_img, output_path, thresh_min = 1, thresh_max=5, step=1):
    """Compares reconstructed image to the reference.
    Pixels for which difference between reconstruction and reference is above specified threshold
    are considered to be anomaly pixels.
    Saves image with anomaly parts highlighted in gradient red. The higher the difference, the brighter the red.
    """
    diff = (reconstruction_img-reference_img)*255
    ref_RGB = cv2.cvtColor(reference_img.squeeze(), cv2.COLOR_GRAY2BGR)*255
    for value in range(thresh_min, thresh_max, step):
        mask = (diff > value).squeeze()
        ref_RGB[mask] += [0,0,255*np.exp((value-thresh_max))]
    cv2.imwrite(output_path, ref_RGB)
