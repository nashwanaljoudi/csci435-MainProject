# ASL Sign Language Recognition — CSCI435 Project

Real-time American Sign Language (ASL) letter recognition using classical computer vision and a trained Random Forest classifier.

University of Wollongong in Dubai — Spring 2026

## How It Works

The pipeline processes each webcam frame through 7 stages:

1. **Enhance** — CLAHE contrast enhancement
2. **Segment** — Skin segmentation via HSV masking
3. **Contour** — Hand contour and bounding box detection
4. **Landmarks** — MediaPipe hand landmark extraction
5. **Background** — Motion detection via MOG2 background subtraction
6. **Classify** — ASL letter prediction (Random Forest)
7. **Smooth** — Temporal smoothing over a rolling window, then word building

Letters commit only when one sign dominates a 25-frame window at 75%+ confidence. The same letter can be signed again after a brief pause (hand leaves the frame).

## Project Structure

```
csci435-MainProject/
├── requirements.txt
├── run_server.py              # Start the web app
├── train_model.py             # Train the Random Forest classifier
├── process_dataset.py         # Prepare training data from raw images
├── cv/                        # CV pipeline
│   ├── 01_enhance.py
│   ├── 02_segment.py
│   ├── 03_contour.py
│   ├── 04_landmarks.py
│   ├── 05_background.py
│   ├── 06_classify.py
│   ├── 07_smooth.py
│   ├── hand_utils.py
│   └── pipeline.py            # Combines all 7 tasks → process_frame()
├── model/                     # Saved model files (.pkl, .task)
├── data/                      # Training/test datasets
├── backend/
│   └── server.py              # FastAPI server (API + static files)
├── frontend/
│   ├── index.html
│   └── static/
└── docs/
```

## Setup

1. **Clone the repo**
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

4. **Run the web app**
   ```bash
   python run_server.py
   ```
   Open **http://localhost:8000** in your browser.

## Using the App

- Click **Start Camera** and sign ASL letters in front of your webcam
- The detected letter shows live as you sign; it commits to the word once held steady
- Click **End Word** to push the current word into your sentence
- Click **Speak** to hear the text read aloud (browser text-to-speech)
- Use **Backspace** to delete the last letter, or sign `DEL`

## Training the Model

```bash
python process_dataset.py   # extract landmarks from raw images
python train_model.py       # train and save classifier to model/
```

## Notes

- Model files go in `model/` — do **not** commit large `.pkl` files or datasets to Git.
- Raw datasets go in `data/` — use `.gitignore` or Git LFS for large files.
