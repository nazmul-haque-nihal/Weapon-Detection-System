import cv2
import numpy as np

def preprocess_image(image, target_size=(640, 640)):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((*target_size, 3), 114, dtype=np.uint8)
    padded[(target_size[1]-new_h)//2:(target_size[1]-new_h)//2+new_h,
           (target_size[0]-new_w)//2:(target_size[0]-new_w)//2+new_w] = resized
    return padded