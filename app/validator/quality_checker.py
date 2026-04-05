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

    def detect_text(self, image: np.ndarray) -> tuple[bool, str, float]:
        """
        Detect if the image contains too much text or looks like a document/ID card.
        Uses MSER (Maximally Stable Extremal Regions) to detect letter-like blobs.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # MSER for text-like blob detection
        mser = cv2.MSER_create(min_area=10, max_area=300)
        regions, _ = mser.detectRegions(gray)
        
        # Calculate metric: num_regions per image area
        height, width = image.shape[:2]
        density = len(regions) / (width * height) * 1000  # Scaling factor
        
        # Normal profile photos usually have < 0.5 density
        # Documents/cards often have > 1.5 density
        is_document = density > 1.2
        
        msg = f"Document-like patterns detected (Text Density: {density:.2f})" if is_document else f"No excessive text detected ({density:.2f})"
        return is_document, msg, density
