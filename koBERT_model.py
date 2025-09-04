import torch
from kobert_tokenizer import KoBERTTokenizer
from pydantic import BaseModel
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooler = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        out = self.dropout(pooler)
        return self.classifier(out)

class DiaryRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotion_type: str

# -----------------------------
# 호출 시 초기화되는 함수
# -----------------------------
def initialize_model(model_path="bert_emotion_weights.pt",
                     csv_path="dataset/AI_hub_emotion_1.csv",
                     label_path="dataset/emotion_label.txt"):

    device = torch.device("cpu")

    # CSV와 레이블 로딩
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    label_dict = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            code, sub, main = [x.strip() for x in line.split(" - ")]
            label_dict[code] = {"sub": sub, "main": main}

    code_to_idx = {code: idx for idx, code in enumerate(label_dict.keys())}
    idx_to_code = {v: k for k, v in code_to_idx.items()}
    LABELS = list(code_to_idx.keys())

    # 모델 로드
    bert_model = BertModel.from_pretrained("skt/kobert-base-v1", return_dict=False)
    model = BERTClassifier(bert_model, num_classes=len(LABELS))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 토크나이저
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    return model, tokenizer, device, LABELS

# -----------------------------
# 감정 분석 함수
# -----------------------------
def analyze_emotion(request: DiaryRequest, model=None, tokenizer=None, device=None, LABELS=None):
    # 모델이 초기화되지 않았다면 초기화
    if model is None or tokenizer is None or device is None or LABELS is None:
        model, tokenizer, device, LABELS = initialize_model()

    text = request.text
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"])
        pred_id = torch.argmax(logits, dim=-1).item()
        pred_label = LABELS[pred_id]

    return {"emotion_type": pred_label}