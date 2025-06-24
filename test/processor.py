import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size

    def load_and_preprocess_images(self, image_paths):
        images = []
        for path in image_paths:
            image = cv2.imread(path)
            if image is not None:
                image = self.resize_and_normalize(image)
                images.append(image)
        return np.array(images)

    def resize_and_normalize(self, image):
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        return image

    def denormalize_images(self, image):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image
