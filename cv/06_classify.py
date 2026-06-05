
# Task 6 - ASL Letter Classification
# Loads a trained classifier and predicts the ASL letter (A-Z)
# from the 63-dimensional MediaPipe landmark feature vector.
#
# TODO: Decide on classifier approach — two options:
# Option A (Recommended): Train our own Random Forest on the Kaggle ASL dataset
#   - Run process_dataset.py to extract landmarks from Kaggle images
#   - Run train_model.py to train and save model/classifier.pkl
#   - Satisfies "trained on custom data" project requirement
#
# Option B: Download a pre-trained ASL landmark classifier from Kaggle
#   - Search: "ASL mediapipe landmarks classifier sklearn"
#   - Must use 63 landmark numbers as input (not raw images)
#   - Save downloaded model as model/classifier.pkl
#   - NOTE: may not fully satisfy "trained on custom data" requirement
#
# Either way, classify.py stays the same — just load model/classifier.pkl and predict