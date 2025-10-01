def postprocess_detections(detections, original_size, input_size=(640, 640)):
    """Convert normalized coordinates to original image coordinates"""
    # YOLOv5 outputs are already in pixel coordinates
    return detections