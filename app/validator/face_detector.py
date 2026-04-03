import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, face_analysis=None):
        # Allow passing a shared InsightFace instance
        if face_analysis:
            self.face_analysis = face_analysis
        else:
            self.face_analysis = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.logger = logging.getLogger(__name__)

    def detect_faces(self, image: np.ndarray):
        """
        Detects faces in an image using InsightFace.
        Returns a list of face objects.
        """
        try:
            faces = self.face_analysis.get(image)
            return faces
        except Exception as e:
            self.logger.error(f"InsightFace detection failed: {e}")
            return []

    def get_face_landmarks(self, image: np.ndarray):
        """
        Get face landmarks for face centering and orientation checks.
        InsightFace returns kps (keypoints) for each face.
        """
        faces = self.detect_faces(image)
        if not faces:
            return []
        return [face.kps for face in faces]

    def get_face_count(self, image: np.ndarray) -> int:
        faces = self.detect_faces(image)
        return len(faces)
