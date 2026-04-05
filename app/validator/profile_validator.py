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
        
        # Track individual criteria for detailed feedback
        criteria = {
            "resolution": {"label": "Adequate Resolution", "status": "pending"},
            "face_count": {"label": "Single Face Detected", "status": "pending"},
            "blur": {"label": "Image Sharpness", "status": "pending"},
            "brightness": {"label": "Proper Lighting", "status": "pending"},
            "framing": {"label": "Professional Framing", "status": "pending"},
            "centering": {"label": "Face Centered", "status": "pending"},
            "human": {"label": "Human Verification", "status": "pending"},
            "orientation": {"label": "Correct Orientation", "status": "pending"},
            "document": {"label": "No Document Text", "status": "pending"}
        }
        
        # 1. Resolution Check (Priority)
        res_ok, res_msg = self.quality_checker.check_resolution(image)
        if not res_ok:
            criteria["resolution"]["status"] = "fail"
            reasons.append(res_msg)
            status = "not_suitable"
        else:
            criteria["resolution"]["status"] = "pass"
            score += 10
            reasons.append("Resolution is acceptable")

        # 2. Face Detection
        faces = self.face_detector.detect_faces(image)
        face_count = len(faces)
        
        if face_count == 0:
            criteria["face_count"]["status"] = "fail"
            criteria["face_count"]["label"] = "No Face Detected"
            reasons.append("No face detected")
            status = "not_suitable"
            return self._build_response(status, score, reasons, warnings, criteria)
        
        if face_count > 1:
            criteria["face_count"]["status"] = "fail"
            criteria["face_count"]["label"] = f"{face_count} Faces Detected"
            reasons.append(f"Multiple faces detected ({face_count})")
            status = "not_suitable"
            # Proceed to score others for partial feedback
        else:
            criteria["face_count"]["status"] = "pass"
            score += 20
            reasons.append("Single face detected")

        # 3. Blur Detection
        is_blurry, blur_val = self.quality_checker.is_blurry(image)
        if is_blurry:
            criteria["blur"]["status"] = "fail"
            criteria["blur"]["label"] = "Image Too Blurry"
            warnings.append(f"Image is slightly blurry")
            status = "not_suitable"
        else:
            criteria["blur"]["status"] = "pass"
            score += 10
            reasons.append("Image is clear")

        # 4. Brightness Detection
        is_dark, brightness_val = self.quality_checker.get_brightness(image)
        if is_dark:
            criteria["brightness"]["status"] = "fail"
            criteria["brightness"]["label"] = "Low Brightness"
            warnings.append(f"Image is too dark")
            status = "not_suitable"
        else:
            criteria["brightness"]["status"] = "pass"
            score += 10
            reasons.append("Lighting is adequate")

        # 5. Selfie and Centering Checks
        is_selfie, selfie_msg = self.selfie_detector.is_selfie(image, faces)
        if is_selfie:
            criteria["framing"]["status"] = "fail"
            criteria["framing"]["label"] = "Background/Selfie Mode"
            warnings.append(selfie_msg)
            status = "not_suitable"
        else:
            criteria["framing"]["status"] = "pass"
            score += 15
            reasons.append("Professional framing (not a close-up selfie)")

        is_centered, center_msg = self.selfie_detector.is_centered(image, faces)
        if not is_centered:
            criteria["centering"]["status"] = "fail"
            criteria["centering"]["label"] = "Face Not Centered"
            warnings.append("Face is not centered in the image")
        else:
            criteria["centering"]["status"] = "pass"
            score += 5 # Additional points for centering
            reasons.append("Face is centered")

        # 6. Human face and orientation
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            tmp_path = tmp_file.name

        try:
            # Human check
            is_human_face, human_msg = self.human_checker.is_human(tmp_path)
            if not is_human_face:
                criteria["human"]["status"] = "fail"
                criteria["human"]["label"] = "Not a Real Person"
                reasons.append(human_msg)
                status = "not_suitable"
            else:
                criteria["human"]["status"] = "pass"
                score += 20
                reasons.append("Verified human face")

            # Orientation check
            is_oriented, orient_msg, _ = self.human_checker.check_orientation(image)
            if not is_oriented:
                criteria["orientation"]["status"] = "fail"
                criteria["orientation"]["label"] = "Improper Pose"
                warnings.append(orient_msg)
                status = "not_suitable"
            else:
                criteria["orientation"]["status"] = "pass"
                score += 10
                reasons.append("Professional pose/orientation")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 7. Document/Text Detection (Penalize if it looks like an ID card/doc)
        is_document, doc_msg, _ = self.quality_checker.detect_text(image)
        if is_document:
            criteria["document"]["status"] = "fail"
            criteria["document"]["label"] = "Document Detected"
            warnings.append(doc_msg)
            status = "not_suitable"
            score -= 50 # Significant penalty for uploading a document
        else:
            criteria["document"]["status"] = "pass"
            score += 10
            reasons.append("No excessive text detected")

        # Final Score adjustment for suitability
        if status == "suitable" and score < 70:
             status = "not_suitable"
             reasons.append("Low overall suitability score")

        return self._build_response(status, score, reasons, warnings, criteria)

    def _build_response(self, status, score, reasons, warnings, criteria=None):
        return {
            "status": status,
            "score": min(score, 100),
            "reasons": reasons,
            "warnings": warnings,
            "criteria": criteria,
            "description": "Image is not suitable for a professional profile." if status == "not_suitable" else "Image is suitable for a professional profile."
        }
