
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. โหลดโมเดลและ Encoder
model = pickle.load(open('model_deployment.pkl', 'rb'))
le_edu = pickle.load(open('encoder_education.pkl', 'rb'))
le_mar = pickle.load(open('encoder_marital.pkl', 'rb'))

st.title("🎯 Customer Response Prediction App")
st.write("แอปพลิเคชันทำนายการตอบรับแคมเปญการตลาด")

# 2. ส่วนรับข้อมูล Input (ให้ตรงกับ 9 Features ของเรา)
col1, col2 = st.columns(2)

with col1:
    education = st.selectbox("ระดับการศึกษา", le_edu.classes_)
    marital = st.selectbox("สถานะสมรส", le_mar.classes_)
    age = st.number_input("อายุ", 18, 100, 30)
    teenhome = st.number_input("จำนวนลูกวัยรุ่นในบ้าน", 0, 5, 0)
    recency = st.number_input("จำนวนวันที่ซื้อครั้งล่าสุด", 0, 100, 10)

with col2:
    num_cat = st.number_input("จำนวนการซื้อผ่าน Catalog", 0, 50, 0)
    num_store = st.number_input("จำนวนการซื้อหน้าร้าน", 0, 50, 0)
    num_web = st.number_input("จำนวนการเข้าชมเว็บไซต์ต่อเดือน", 0, 50, 0)
    total_promo = st.number_input("จำนวนโปรโมชั่นที่เคยตอบรับ", 0, 10, 0)

# 3. ส่วนการทำนาย
if st.button("ทำนายผล"):
    # แปลงค่า Categorical
    edu_encoded = le_edu.transform([education])[0]
    mar_encoded = le_mar.transform([marital])[0]
    
    # รวมข้อมูลเป็น Array
    features = np.array([[edu_encoded, mar_encoded, teenhome, recency, 
                         num_cat, num_store, num_web, age, total_promo]])
    
    prediction = model.predict(features)
    
    st.subheader("ผลการทำนาย:")
    if prediction[0] == 1:
        st.success("✅ ลูกค้าคนนี้ 'มีแนวโน้มตอบรับ' แคมเปญ")
    else:
        st.error("❌ ลูกค้าคนนี้ 'ไม่มีแนวโน้มตอบรับ' แคมเปญ")
    