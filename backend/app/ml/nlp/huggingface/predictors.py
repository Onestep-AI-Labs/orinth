from pathlib import Path

from app.ml.common.base import Predictor
from app.schemas import Detection, InferenceParameters


class HuggingFaceTextClassificationPredictor(Predictor):
    def __init__(self, model_path: Path, labels: list[str] | None = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Hugging Face model not found: {model_path}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.labels = labels or [
            self.model.config.id2label.get(index, f"class_{index}")
            for index in range(int(self.model.config.num_labels))
        ]

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        import torch  # noqa: PLC0415

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=0).tolist()
        scores = {
            label: round(float(probabilities[index]), 6)
            for index, label in enumerate(self.labels)
            if index < len(probabilities)
        }
        label = max(scores.items(), key=lambda item: item[1])[0] if scores else self.labels[0]
        return {"task": "text_classification", "label": label, "scores": scores}


class HuggingFaceSummarizerPredictor(Predictor):
    def __init__(self, model_path: Path, labels: list[str] | None = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Hugging Face model not found: {model_path}")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        import torch  # noqa: PLC0415

        max_length = max(parameters.max_length, 1)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                early_stopping=True,
            )
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"task": "summarization", "summary": summary}


class HuggingFaceQAPredictor(Predictor):
    def __init__(self, model_path: Path, labels: list[str] | None = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Hugging Face model not found: {model_path}")
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.eval()

    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        return []

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        import torch  # noqa: PLC0415

        question = parameters.question or ""
        inputs = self.tokenizer(
            question,
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=384,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            start = int(torch.argmax(outputs.start_logits, dim=-1)[0])
            end = int(torch.argmax(outputs.end_logits, dim=-1)[0])
        if end < start:
            end = start
        answer_ids = inputs["input_ids"][0][start : end + 1]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        start_score = torch.softmax(outputs.start_logits, dim=-1)[0][start]
        end_score = torch.softmax(outputs.end_logits, dim=-1)[0][end]
        score = round(float((start_score * end_score).sqrt()), 6)
        return {"task": "question_answering", "answer": answer, "score": score}
