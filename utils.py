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
    df.columns = df.columns.str.strip().str.replace('"', '')  # Çift tırnak ve boşlukları temizle
    return dict(zip(df["Soru no"], df["Sağlıklı Çocukta Beklenen Cevap"]))

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
    """
    Her bir grup için risk veren ve vermeyen modellerin ağırlıklı toplamlarını hesaplar,
    risk yüzdesini (örneğin %65) ve final kararı (riskli veya risksiz) verir.
    Ayrıca, açıklanabilirlik için, modelde kullanılan sorulardan beklenen cevaptan farklı olanları listeler.
    """
    summary = {}

    # Gruplama: her hastalık/beceri için ilgili modelleri belirliyoruz.
    groups = {
        'Sosyal': [],
        'Duyusal': [],
        'Motor': [],
        'Dil': [],
        'Iletisim': [],
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
        risk_weight_sum = 0.0      # Risk veren modellerin ağırlık toplamı
        nonrisk_weight_sum = 0.0   # Risk vermeyen modellerin ağırlık toplamı
        wrong_questions_all = []   # Açıklama için hatalı cevaplanan sorular

        for model_name in model_names:
            model = models[model_name]
            X = input_data.get(model_name)
            if X is None:
                continue

            # Modelin risk olasılığı
            proba = model.predict_proba(X)[0][1]
            # Basit eşik: 0.5 üzerinde risk, altındaysa risksiz
            binary_pred = 1 if proba >= 0.5 else 0
            # Modelin ağırlığı: train ve test F1 skorlarının ortalaması (performances üzerinden)
            weight = performances.get(model_name, 1.0)

            if binary_pred == 1:
                risk_weight_sum += weight

                # Modelin kullandığı sorular arasında, beklenen cevap ile farklı cevaplanmış olanları topla.
                used_questions = feature_lists.get(model_name, [])
                wrong_questions = [
                    q for q in used_questions
                    if q in expected_answers and answers.get(q) != expected_answers[q]
                ]
                wrong_questions_all.extend(wrong_questions)
            else:
                nonrisk_weight_sum += weight

        total_weight = risk_weight_sum + nonrisk_weight_sum
        # İki ağırlık toplamı üzerinden risk yüzdesini hesapla
        risk_percentage = (risk_weight_sum / total_weight * 100) if total_weight > 0 else 0

        # Final karar: risk veren modellerin ağırlığı daha yüksekse, riskli olarak belirle.
        final_pred = 1 if risk_weight_sum > nonrisk_weight_sum else 0

        # Soru havuzu: tüm modellerin birleşik soru listesi
        combined_question_pool = set()
        for model_name in model_names:
            combined_question_pool.update(feature_lists.get(model_name, []))
        
        # Beklenenden farklı yanıtlanan sorular
        incorrect_answers_detailed = []
        for q in combined_question_pool:
            expected = expected_answers.get(q)
            given = answers.get(q)
            if expected and given and expected != given:
                incorrect_answers_detailed.append({
                    "soru_kodu": q,
                    "soru": q,  # metin app.py'de yazdırılırken eklenebilir
                    "beklenen": expected,
                    "verilen": given
                })
        
        summary[label] = {
            "risk_weight_sum": risk_weight_sum,
            "nonrisk_weight_sum": nonrisk_weight_sum,
            "risk_percentage": risk_percentage,
            "final_prediction": final_pred,
            "wrong_questions": list(set(wrong_questions_all)),
            "total_models": len(model_names),
            "total_positive": sum(1 for model_name in model_names
                                  if model_name in input_data and
                                     models[model_name].predict_proba(input_data[model_name])[0][1] >= 0.5),
            "incorrect_answers_detailed": incorrect_answers_detailed
        }

    return summary

