import jiwer
import os
from app.services.asr_service import transcribe_audio

SAMPLES_DIR = "samples"

references = []
predictions = []

# Load ground truth
with open(os.path.join(SAMPLES_DIR, "ground_truth.txt"), "r") as f:
    lines = f.readlines()

for line in lines:
    filename, ground_truth = line.strip().split("|")

    file_path = os.path.join(SAMPLES_DIR, filename)

    class DummyFile:
        def __init__(self, path):
            self.file = open(path, "rb")

    file = DummyFile(file_path)

    predicted_text = transcribe_audio(file)

    references.append(ground_truth.lower())
    predictions.append(predicted_text.lower())

wer = jiwer.wer(references, predictions)

print("\nREFERENCES:", references)
print("\nPREDICTIONS:", predictions)
print(f"\n✅ Word Error Rate (WER): {wer:.2f}")
