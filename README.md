# ASL Sign Language Recognition — CSCI435 Project

Real-time American Sign Language (ASL) letter recognition using classical computer vision techniques and a custom-trained Random Forest classifier.

University of Wollongong in Dubai — Spring 2026

## Team Structure

| Team | Owns |
|------|------|
| CV Team | `cv/` — all 7 pipeline tasks |
| Website Team | `frontend/` — Streamlit app, webcam, upload |
| Report Team | `docs/` — written report |

## Project Structure

```
csci435-MainProject/
├── requirements.txt
├── README.md
├── .gitignore
├── test_pipeline.py          # Run the CV pipeline without the website
├── cv/                       # CV pipeline (CV Team)
│   ├── enhance.py            # Task 1 - Image Enhancement (CLAHE)
│   ├── segment.py            # Task 2 - Skin Segmentation (HSV + morphology)
│   ├── contour.py            # Task 3 - Hand Contour & Bounding Box
│   ├── landmarks.py          # Task 4 - MediaPipe landmark extraction
│   ├── background.py         # Task 5 - Background Modelling (MOG2)
│   ├── classify.py           # Task 6 - ASL Classification (Random Forest)
│   ├── smooth.py             # Task 7 - Temporal Smoothing & Word Builder
│   └── pipeline.py           # Combines all tasks → process_frame(frame)
├── model/                    # Saved trained model files (.pkl, etc.)
├── data/                     # Training/test datasets
├── frontend/                 # Streamlit web app (Website Team)
│   ├── app.py
│   ├── webcam.py
│   └── upload.py
└── docs/                     # Project report (Report Team)
    └── report.pdf
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd csci435-MainProject
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the CV pipeline (no browser needed)**
   ```bash
   python test_pipeline.py
   ```

5. **Run the full web app**
   ```bash
   streamlit run frontend/app.py
   ```

## Notes

- Trained model files go in `model/` — do **not** commit large dataset files to Git.
- Raw datasets go in `data/` — add large files to `.gitignore` or use Git LFS.
