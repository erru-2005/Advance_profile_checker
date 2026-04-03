import cv2
import numpy as np

class QualityChecker:
    def __init__(self, blur_threshold=100, brightness_threshold=50):
        self.blur_threshold = blur_threshold
        self.brightness_threshold = brightness_threshold

    def is_blurry(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Check if the image is blurry using the Laplacian variance method.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold, variance

    def get_brightness(self, image: np.ndarray) -> tuple[bool, float]:
        """
        Check if the image is too dark.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        mean_brightness = np.mean(v_channel)
        return mean_brightness < self.brightness_threshold, mean_brightness

    def check_resolution(self, image: np.ndarray, min_width=300, min_height=300) -> tuple[bool, str]:
        """
        Check if the image resolution meets the minimum requirements.
        """
        height, width = image.shape[:2]
        if width < min_width or height < min_height:
            return False, f"Resolution too low: {width}x{height} (min {min_width}x{min_height})"
        return True, f"Resolution ok: {width}x{height}"
