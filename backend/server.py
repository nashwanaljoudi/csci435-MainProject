"""
FastAPI backend for the ASL sign language detector.
Serves the web frontend and exposes endpoints for frame processing and text controls.
Run: uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
"""

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cv.pipeline import (
    backspace,
    finalize_word,
    get_text_state,
    process_frame,
    reset_state,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

app = FastAPI(title="ASL Sign Language Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameRequest(BaseModel):
    image: str
    require_motion: bool = False


class ProcessResponse(BaseModel):
    annotated_image: str
    letter: str | None
    committed: str | None
    confidence: float | None
    word: str
    sentence: str
    full_text: str


class TextResponse(BaseModel):
    word: str
    sentence: str
    full_text: str


def _decode_image(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


def _encode_image(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated frame")
    return base64.b64encode(buf).decode("ascii")


@app.get("/api/health")
def health():
    model_path = ROOT / "model" / "classifier.pkl"
    return {
        "status": "ok",
        "model_loaded": model_path.exists(),
    }


@app.post("/api/process-frame", response_model=ProcessResponse)
def api_process_frame(body: FrameRequest):
    frame = _decode_image(body.image)
    annotated, committed, confidence, letter, word, sentence = process_frame(
        frame, require_motion=body.require_motion
    )

    state = get_text_state()
    return ProcessResponse(
        annotated_image=_encode_image(annotated),
        letter=letter,
        committed=committed,
        confidence=confidence,
        word=word,
        sentence=sentence,
        full_text=state["full_text"],
    )


@app.post("/api/reset", response_model=TextResponse)
def api_reset():
    reset_state()
    return TextResponse(word="", sentence="", full_text="")


@app.post("/api/finalize-word", response_model=TextResponse)
def api_finalize_word():
    finalize_word()
    return TextResponse(**get_text_state())


@app.post("/api/backspace", response_model=TextResponse)
def api_backspace():
    backspace()
    return TextResponse(**get_text_state())


@app.get("/api/text", response_model=TextResponse)
def api_text():
    state = get_text_state()
    return TextResponse(**state)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if (FRONTEND / "static").exists():
    app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")
