# Faculty Registration Portal 🎓

A modern, professional-grade Faculty Enrollment system built with **FastAPI** and **AI-powered Image Validation**. This portal ensures high data integrity with real-time validation, automatic salutation detection, and intelligent form auto-filling for existing faculty members.

## 🌟 Key Features

*   **AI Profile Guard**: Uses `InsightFace` and `DeepFace` to verify professional photograph suitability (Face detection, orientation, and human-only checks).
*   **Intelligent Form**: 
    *   **Auto-Fill**: Instantly loads existing details when a registered Faculty ID is entered.
    *   **Auto-Formatting**: Automatic **Title Case** conversion for names (e.g., "mr. errol" → "Mr. Errol").
    *   **Real-time Validation**: Glowing visual cues (Green/Red) and matching icons for all fields.
*   **Zero-Dependency Design**: Custom hardcoded SVG system ensures icons work offline/without CDNs.
*   **Interactive UX**: "Jump-to-Error" logic on submission and localized field messages.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8 or higher
- Git

### 2. Set up Virtual Environment
It is highly recommended to run this in a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/MacOS)
source venv/bin/activate
```

### 3. Installation
Install the project dependencies:
```bash
pip install -r app/requirements.txt
```

### 4. Running the Application

Start the server using Uvicorn:
```bash
uvicorn app.main:app --reload
```
The application will be accessible at: `http://127.0.0.1:8000`

## 📁 Project Structure

*   `app/main.py`: Core FastAPI backend and AI endpoints.
*   `app/templates/index.html`: Modern, responsive frontend built with Vanilla CSS/JS.
*   `app/static/`: Secure storage for verified and unverified faculty photographs.
*   `app/faculties.json`: Local persistent storage for faculty records.

## 🛡️ Security & Integrity

- **ID Format**: Enforces strict `BBHCF...` serialization.
- **Domain Lock**: Only official `@bbhegdecollege.com` emails are accepted.
- **AI Verification**: Prevents non-human or unsuitable profiles from being uploaded to the system index.

---
Developed for streamlined academic directory management.
