import numpy as np
import cv2

class SelfieDetector:
    def __init__(self, face_size_threshold=0.6):
        self.face_size_threshold = face_size_threshold

    def is_selfie(self, image: np.ndarray, detections) -> tuple[bool, str]:
        """
        Check if the image is a selfie based on face size and framing.
        """
        if not detections:
            return False, "No face detected"

        height, width = image.shape[:2]
        
        # InsightFace returns detections as face objects with a .bbox attribute [x1, y1, x2, y2]
        face = detections[0]
        x1, y1, x2, y2 = face.bbox
        
        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        image_area = width * height
        
        face_percentage = face_area / image_area
        
        # 1. Face size check
        if face_percentage > self.face_size_threshold:
            return True, f"Face occupies {face_percentage:.2%} of the image, which is higher than the {self.face_size_threshold:.0%} threshold (close-up/selfie)."

        # 2. Framing check (Close-up framing)
        # Using relative coordinates for consistency with previous logic
        rel_x1, rel_y1 = x1 / width, y1 / height
        rel_x2, rel_y2 = x2 / width, y2 / height
        
        if rel_x1 < 0.05 or rel_y1 < 0.05 or rel_x2 > 0.95 or rel_y2 > 0.95:
             if face_percentage > 0.45:
                 return True, "Close-up framing detected (face occupies significant portion and is near edges)."

        return False, "Not a selfie"

    def is_centered(self, image: np.ndarray, detections) -> tuple[bool, str]:
        """
        Check if the face is centered in the image.
        """
        if not detections:
             return False, "No face detected"
             
        height, width = image.shape[:2]
        # InsightFace bbox format [x1, y1, x2, y2]
        face = detections[0]
        x1, y1, x2, y2 = face.bbox
        
        face_center_x = (x1 + x2) / (2 * width)
        face_center_y = (y1 + y2) / (2 * height)
        
        # Allow 15% deviation from center
        if 0.35 < face_center_x < 0.65 and 0.35 < face_center_y < 0.65:
            return True, "Face is centered"
        
        return False, "Face is not centered"
