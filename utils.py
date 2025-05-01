import os
import joblib
import pandas as pd
import numpy as np
import csv
from typing import Dict, List, Any
from config import MODEL_LIST, MODEL_DIR, FEATURE_FILE, PERFORMANCE_FILE

def load_models() -> Dict[str, Any]:
    models = {}
    for model_name in MODEL_LIST:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    return models

def load_feature_lists(feature_file: str = FEATURE_FILE) -> Dict[str, List[str]]:
    df = pd.read_excel(feature_file)
    return {
        row["Model"]: [q.strip() for q in str(row["Selected_Questions"]).split(",") if q.strip()]
        for _, row in df.iterrows()
    }

def load_model_performances(performance_file: str = PERFORMANCE_FILE) -> Dict[str, float]:
    df = pd.read_excel(performance_file)
    return {
        row["Model"]: (row.get("Train_F1", 0) + row.get("Test_F1", 0)) / 2
        for _, row in df.iterrows()
    }

def load_expected_answers(csv_file: str = "SorularFull.csv") -> Dict[str, str]:
    df = pd.read_csv(
        csv_file,
        sep=';',
        encoding='windows-1254',
        engine='python',
        quoting=csv.QUOTE_NONE,
        quotechar=None,
        escapechar='\\'
    )
    return dict(zip(df["SoruNo"], df["Sağlıklı Çocukta Beklenen Cevap"]))

def prepare_input_data(answers: Dict[str, str], feature_lists: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    input_data = {}
    for model_name, features in feature_lists.items():
        row = []
        for f in features:
            row.append(1 if answers.get(f) == "Evet" else 0)
        input_data[model_name] = np.array(row).reshape(1, -1)
    return input_data

def make_predictions(
    models: Dict[str, Any],
    input_data: Dict[str, np.ndarray],
    performances: Dict[str, float],
    feature_lists: Dict[str, List[str]],
    answers: Dict[str, str],
    expected_answers: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:

    summary = {}

    groups = {
        'Sosyal': [],
        'Duyusal': [],
        'Motor': [],
        'Dil': [],
        'İletişim': [],
        'Ortak_Dikkat': [],
        'Otizm': [],
        'DEHB': [],
        'Dil ve Konuşma Bozuklukları': [],
        'Gelişimsel Koordinasyon Bozukluğu': [],
        'Zihinsel Yetersizlik': []
    }

    for model_name in models:
        for key in groups:
            if model_name.endswith(key):
                groups[key].append(model_name)

    for label, model_names in groups.items():
        predictions = []
        weights = []
        all_wrong_questions = []

        for model_name in model_names:
            model = models[model_name]
            X = input_data.get(model_name)
            if X is None:
                continue
            y_pred = model.predict_proba(X)[0][1]
            binary_pred = 1 if y_pred >= 0.5 else 0
            weight = performances.get(model_name, 1.0)

            predictions.append(binary_pred * weight)
            weights.append(weight)

            used_questions = feature_lists.get(model_name, [])
            wrong_questions = [
                q for q in used_questions
                if q in expected_answers and answers.get(q) != expected_answers[q]
            ]
            if binary_pred == 1:
                all_wrong_questions.extend(wrong_questions)

        if weights:
            weighted_score = sum(predictions) / sum(weights)
            total_models = len(weights)
            total_positive = sum([1 for p in predictions if p > 0])
            final_pred = 1 if weighted_score >= 0.5 else 0

            summary[label] = {
                "total_models": total_models,
                "total_positive": total_positive,
                "final_prediction": final_pred,
                "weighted_score": weighted_score,
                "wrong_questions": list(set(all_wrong_questions))
            }

    return summary

