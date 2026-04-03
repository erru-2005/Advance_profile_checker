import cv2
import numpy as np
import os
import tempfile
from insightface.app import FaceAnalysis
from .face_detector import FaceDetector
from .quality_checker import QualityChecker
from .selfie_detector import SelfieDetector
from .human_checker import HumanChecker

class ProfileValidator:
    def __init__(self):
        # Initialize InsightFace once for all components
        self.face_analysis = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        
        self.face_detector = FaceDetector(face_analysis=self.face_analysis)
        self.quality_checker = QualityChecker()
        self.selfie_detector = SelfieDetector()
        self.human_checker = HumanChecker(face_analysis=self.face_analysis)

    def validate(self, image: np.ndarray) -> dict:
        reasons = []
        warnings = []
        score = 0
        status = "suitable"
        
        # 1. Resolution Check (Priority)
        res_ok, res_msg = self.quality_checker.check_resolution(image)
        if not res_ok:
            reasons.append(res_msg)
            status = "not_suitable"
        else:
            score += 10
            reasons.append("Resolution is acceptable")

        # 2. Face Detection
        faces = self.face_detector.detect_faces(image)
        face_count = len(faces)
        
        if face_count == 0:
            reasons.append("No face detected")
            status = "not_suitable"
            return self._build_response(status, score, reasons, warnings)
        
        if face_count > 1:
            reasons.append(f"Multiple faces detected ({face_count})")
            status = "not_suitable"
            # Proceed to score others for partial feedback
        else:
            score += 20
            reasons.append("Single face detected")

        # 3. Blur Detection
        is_blurry, blur_val = self.quality_checker.is_blurry(image)
        if is_blurry:
            warnings.append(f"Image is slightly blurry (Laplacian: {blur_val:.2f})")
            status = "not_suitable"
        else:
            score += 10
            reasons.append("Image is clear")

        # 4. Brightness Detection
        is_dark, brightness_val = self.quality_checker.get_brightness(image)
        if is_dark:
            warnings.append(f"Image is too dark (Brightness: {brightness_val:.2f})")
            status = "not_suitable"
        else:
            score += 10
            reasons.append("Lighting is adequate")

        # 5. Selfie and Centering Checks
        is_selfie, selfie_msg = self.selfie_detector.is_selfie(image, faces)
        if is_selfie:
            warnings.append(selfie_msg)
            status = "not_suitable"
        else:
            score += 15
            reasons.append("Professional framing (not a close-up selfie)")

        is_centered, center_msg = self.selfie_detector.is_centered(image, faces)
        if not is_centered:
            warnings.append("Face is not centered in the image")
        else:
            score += 5 # Additional points for centering
            reasons.append("Face is centered")

        # 6. Human face and orientation (requires DeepFace/InsightFace)
        # DeepFace needs a path or array. Let's use a temp file for DeepFace path preference.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            tmp_path = tmp_file.name

        try:
            # Human check
            is_human_face, human_msg = self.human_checker.is_human(tmp_path)
            if not is_human_face:
                reasons.append(human_msg)
                status = "not_suitable"
            else:
                score += 20
                reasons.append("Verified human face")

            # Orientation check
            is_oriented, orient_msg, _ = self.human_checker.check_orientation(image)
            if not is_oriented:
                warnings.append(orient_msg)
                status = "not_suitable"
            else:
                score += 10
                reasons.append("Professional pose/orientation")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Final Score adjustment for suitability
        if status == "suitable" and score < 70:
             status = "not_suitable"
             reasons.append("Low overall suitability score")

        return self._build_response(status, score, reasons, warnings)

    def _build_response(self, status, score, reasons, warnings):
        return {
            "status": status,
            "score": min(score, 100),
            "reasons": reasons,
            "warnings": warnings,
            "description": "Image is not suitable for a professional profile." if status == "not_suitable" else "Image is suitable for a professional profile."
        }
