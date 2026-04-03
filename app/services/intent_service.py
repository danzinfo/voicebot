from transformers import BertTokenizer, BertForSequenceClassification
from app.config import INTENT_MODEL_PATH
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained(INTENT_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
model.eval()

id2label = model.config.id2label

def predict_intent(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
    return id2label[pred.item()], confidence.item()



##✔ Loads your trained model (models/intent_model)
##✔ Uses model’s own label mapping (id2label)
##✔ Removed hardcoded labels (no mismatch risk)
##✔ Uses model.eval() (correct inference mode)
##✔ Uses torch.no_grad() (efficient + safe)
