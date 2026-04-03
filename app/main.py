from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import io
import os
from PIL import Image
from .validator.profile_validator import ProfileValidator
import logging

# Get the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Profile Image Validation API",
    description="Analyze and validate professional profile images using AI.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize Validator
# Note: Initializing this at startup to load models once
validator = ProfileValidator()

class ValidationResponse(BaseModel):
    status: str
    score: int
    reasons: list[str]
    warnings: list[str]
    description: str

class FacultyDetails(BaseModel):
    faculty_id: str
    full_name: str
    gender: str
    address: str
    college_email: str
    personal_email: str
    whatsapp_number: str
    profile_image_url: str = "" # In a real app we'd save the file path
    # Additional data captured during validation
    image_suitability_score: int = 0
    image_suitability_status: str = "unknown"

@app.get("/")
async def root(request: Request):
    # Using keyword arguments to be compatible with all Starlette versions
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

# Silent handlers for browser noise to keep logs clean
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(content="", status_code=204)

@app.get("/.well-known/{path:path}", include_in_schema=False)
async def well_known(path: str):
    return HTMLResponse(content="", status_code=404)

@app.post("/validate-profile-image", response_model=ValidationResponse)
async def validate_profile_image(file: UploadFile = File(...), faculty_id: str | None = None):
    """
    Upload an image for professional profile validation.
    Supported formats: jpg, jpeg, png, webp.
    If faculty_id is provided, saves the image to appropriate folders.
    """
    logger.info(f"Received validation request for file: {file.filename} (Type: {file.content_type})")
    
    # Relaxed validation: check content type or file extension
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    is_valid_type = file.content_type in allowed_types
    is_valid_ext = any(file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"])
    
    if not (is_valid_type or is_valid_ext):
        logger.warning(f"Rejected invalid file type: {file.content_type} for file {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload a JPG, PNG, or WEBP image."
        )

    try:
        # Read image file
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents))
        
        # Get extension
        ext = os.path.splitext(file.filename)[1].lower()
        if not ext:
            ext = ".jpg" # Default
        
        # Convert to RGB if necessary
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
            
        # Convert to numpy array for OpenCV (BGR)
        open_cv_image = np.array(image_pil)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Run validation
        result = validator.validate(open_cv_image)
        
        # Save image if faculty_id is provided
        if faculty_id:
            if result['status'] == 'suitable' and result['score'] >= 100:
                # Save to faculty_pictures
                save_dir = os.path.join(STATIC_DIR, "faculty_pictures")
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{faculty_id}{ext}"
                save_path = os.path.join(save_dir, filename)
                image_pil.save(save_path)
                logger.info(f"Saved suitable image for {faculty_id} at {save_path}")
            else:
                # Save to faculty_picture_error
                save_dir = os.path.join(STATIC_DIR, "faculty_picture_error")
                os.makedirs(save_dir, exist_ok=True)
                
                # Determine error filename
                i = 1
                while True:
                    filename = f"{faculty_id}_error{i}{ext}"
                    if not os.path.exists(os.path.join(save_dir, filename)):
                        break
                    i += 1
                
                save_path = os.path.join(save_dir, filename)
                image_pil.save(save_path)
                logger.info(f"Saved unsuitable image for {faculty_id} at {save_path}")

        return result

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get-faculty/{faculty_id}")
async def get_faculty(faculty_id: str):
    """
    Retrieve faculty details by ID.
    Used for auto-populating the form if a record already exists.
    """
    import json
    file_path = os.path.join(BASE_DIR, "faculties.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Faculty not found")
        
    try:
        with open(file_path, "r") as f:
            faculties = json.load(f)
            
        # Convert search ID to uppercase for matching
        search_id = faculty_id.upper()
        if search_id in faculties:
            return faculties[search_id]
        else:
            raise HTTPException(status_code=404, detail="Faculty not found")
            
    except Exception as e:
        logger.error(f"Error retrieving faculty data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/save-faculty")
async def save_faculty(data: FacultyDetails):
    """
    Save faculty details to a JSON file.
    """
    import json
    file_path = os.path.join(BASE_DIR, "faculties.json")
    
    try:
        # Load existing data
        faculties: dict = {}
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                faculties = json.load(f)
        
        # Save or update faculty record
        faculties[data.faculty_id] = data.dict()
        
        with open(file_path, "w") as f:
            json.dump(faculties, f, indent=4)
            
        logger.info(f"Saved details for Faculty ID: {data.faculty_id}")
        return {"message": "Faculty details saved successfully!", "faculty_id": data.faculty_id}
        
    except Exception as e:
        logger.error(f"Error saving faculty data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")

@app.get("/api-status")
async def api_status():
    return {
        "message": "Welcome to the Profile Image Validation API. Use /docs for documentation.",
        "endpoint": "/validate-profile-image"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
