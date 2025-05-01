# -*- coding: utf-8 -*-
import streamlit as st
from utils import load_models, load_feature_lists, load_model_performances, prepare_input_data, make_predictions
import pandas as pd
import random

# Sayfa yapılandırması
st.set_page_config(
    page_title="Nörogelişimsel Bozukluk Tahmin Sistemi",
    page_icon="🧠",
    layout="wide"
)

# CSS stilleri
st.markdown("""
<style>
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .high-risk {
        background-color: rgba(255, 0, 0, 0.1);
        border: 2px solid red;
    }
    .medium-risk {
        background-color: rgba(255, 165, 0, 0.1);
        border: 2px solid orange;
    }
    .low-risk {
        background-color: rgba(0, 255, 0, 0.1);
        border: 2px solid green;
    }
</style>
""", unsafe_allow_html=True)

# 🔐 Sabit 95 soru listesi
questions = [
    'Q2','Q4','Q8','Q9','Q13','Q14','Q16','Q18','Q19','Q20','Q21','Q25','Q26','Q28','Q29',
    'Q33','Q34','Q35','Q40','Q44','Q45','Q47','Q51','Q52','Q53','Q54','Q60','Q62','Q67',
    'Q71','Q77','Q81','Q82','Q86','Q89','Q93','Q95','Q96','Q105','Q108','Q115','Q116',
    'Q117','Q119','Q125','Q126','Q127','Q128','Q129','Q130','Q133','Q138','Q139','Q140',
    'Q144','Q151','Q158','Q159','Q163','Q166','Q174','Q179','Q184','Q185','Q187','Q192',
    'Q197','Q202','Q203','Q204','Q205','Q210','Q212','Q215','Q219','Q221','Q222','Q224',
    'Q226','Q227','Q229','Q230','Q231','Q232','Q233','Q234','Q235','Q236','Q239','Q241',
    'Q242','Q243','Q249','Q252','Q253'
]

def main():
    st.title("Nörogelişimsel Bozukluk Tahmin Sistemi")
    
    try:
        # Modelleri, özellikleri ve performans metriklerini yükle
        models = load_models()
        feature_lists = load_feature_lists()
        model_performances = load_model_performances()

        # Rastgele Doldur butonu (form dışında)
        if st.button("Rastgele Doldur"):
            for q in questions:
                st.session_state[q] = random.choice(["Evet", "Hayır"])

        # Form oluştur
        with st.form("prediction_form"):
            st.subheader("Lütfen aşağıdaki soruları cevaplayınız:")
            answers = {}

            for q in questions:
                answers[q] = st.radio(
                    label=q,
                    options=["Evet", "Hayır"],
                    key=q,
                    index=0 if st.session_state.get(q) == "Evet" else 1 if st.session_state.get(q) == "Hayır" else 0
                )

            submit_button = st.form_submit_button("Tahmin Yap")
        
        if submit_button:
            # Kullanıcı cevaplarını hazırla
            input_data = prepare_input_data(answers, feature_lists)
            
            # Tahminleri yap
            predictions = make_predictions(models, input_data, model_performances)
            
            st.subheader("Tahmin Sonuçları:")
            
            for condition, prob in predictions.items():
                prob_percentage = prob * 100
                if prob_percentage >= 70:
                    risk_class = "high-risk"
                    risk_level = "Yüksek"
                elif prob_percentage >= 40:
                    risk_class = "medium-risk"
                    risk_level = "Orta"
                else:
                    risk_class = "low-risk"
                    risk_level = "Düşük"
                
                st.markdown(
                    f"""
                    <div class="risk-box {risk_class}">
                        <h4>{condition}</h4>
                        <p>Risk Seviyesi: {risk_level}</p>
                        <p>Risk Oranı: {prob_percentage:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.error("Lütfen tüm gerekli dosyaların mevcut olduğundan emin olun (models klasörü, selected_features.xlsx, model_performance.xlsx)")

if __name__ == "__main__":
    main()
