import cv2
import numpy as np
import logging
from deepface import DeepFace
from insightface.app import FaceAnalysis

class HumanChecker:
    def __init__(self, face_analysis=None):
        # Allow passing a shared InsightFace instance
        if face_analysis:
            self.face_analysis = face_analysis
        else:
            self.face_analysis = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.logger = logging.getLogger(__name__)

    def is_human(self, image_path: str) -> tuple[bool, str]:
        """
        Use DeepFace to verify if the image contains a human face.
        DeepFace will throw an exception if no face is detected or if it is a cartoon.
        """
        try:
            # Analyze image for emotion/age to verify it's a real human face
            # DeepFace.analyze will use its internal models to check for human traits
            objs = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True)
            if objs:
                return True, "Human face verified via DeepFace analysis."
            return False, "Could not verify human face."
        except Exception as e:
            self.logger.error(f"DeepFace analysis failed: {e}")
            return False, f"DeepFace could not identify a human face: {str(e)}"

    def check_orientation(self, image: np.ndarray) -> tuple[bool, str, float]:
        """
        Use InsightFace to detect head tilt (roll).
        Reject if roll > 20 degrees.
        """
        try:
            faces = self.face_analysis.get(image)
            if not faces:
                return False, "No face detected by InsightFace", 0.0
            
            # Use the first face detected
            face = faces[0]
            # pose is (pitch, yaw, roll)
            pitch, yaw, roll = face.pose
            
            if abs(float(roll)) > 20:
                return False, f"Head tilted too much ({abs(float(roll)):.2f} degrees)", float(roll)
            
            return True, f"Head orientation ok (tilt: {abs(float(roll)):.2f} degrees)", float(roll)
        except Exception as e:
            self.logger.error(f"InsightFace orientation check failed: {e}")
            return False, f"Orientation check failed: {str(e)}", 0.0
