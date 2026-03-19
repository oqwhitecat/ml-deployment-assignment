import streamlit as st
import pandas as pd
import pickle
import numpy as np

# โหลดโมเดลและ encoders
@st.cache_resource
def load_model():
    with open('model_deployment.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder_education.pkl', 'rb') as f:
        le_edu = pickle.load(f)
    with open('encoder_marital.pkl', 'rb') as f:
        le_mar = pickle.load(f)
    with open('num_imputer.pkl', 'rb') as f:
        num_imp = pickle.load(f)
    with open('cat_imputer.pkl', 'rb') as f:
        cat_imp = pickle.load(f)
    return model, le_edu, le_mar, num_imp, cat_imp

model, le_edu, le_mar, num_imp, cat_imp = load_model()

st.title("💰 ทำนายการตอบรับแคมเปญ (Customer Response)")
st.write("กรอกข้อมูลลูกค้าเพื่อทำนายว่าเขาจะตอบรับแคมเปญหรือไม่")

# สร้าง input fields
education = st.selectbox("ระดับการศึกษา", ["Unknown", "Graduation", "PhD", "Master", "Basic", "2n Cycle"])
marital_status = st.selectbox("สถานภาพ", ["Unknown", "Married", "Together", "Single", "Divorced", "Widow", "Alone", "Absurd", "YOLO"])
teenhome = st.number_input("จำนวนวัยรุ่นในบ้าน", min_value=0, max_value=2, value=0)
recency = st.number_input("จำนวนวันตั้งแต่ครั้งสุดท้ายที่ซื้อ", min_value=0, value=49)
num_catalog_purchases = st.number_input("จำนวนการซื้อผ่านแคตตาล็อก", min_value=0, value=2)
num_store_purchases = st.number_input("จำนวนการซื้อที่ร้าน", min_value=0, value=5)
num_web_visits_month = st.number_input("จำนวนครั้งที่เข้าชมเว็บต่อเดือน", min_value=0, value=3)
age = st.number_input("อายุ", min_value=18, max_value=120, value=55)
total_promo = st.number_input("จำนวนแคมเปญที่เข้าร่วม", min_value=0, value=1)

# เมื่อกดปุ่มทำนาย
if st.button("ทำนาย"):
    # สร้าง DataFrame จากข้อมูลที่กรอก
    input_data = pd.DataFrame([[education, marital_status, teenhome, recency,
                                num_catalog_purchases, num_store_purchases,
                                num_web_visits_month, age, total_promo]],
                              columns=['Education', 'Marital_Status', 'Teenhome',
                                       'Recency', 'NumCatalogPurchases', 'NumStorePurchases',
                                       'NumWebVisitsMonth', 'Age', 'Total_Promo'])
    
    # จัดการ Missing Values (แม้เราจะกรอกครบ แต่เพื่อความปลอดภัย)
    # ควรใช้ imputer ที่โหลดมา แต่ imputer ต้องการ fit transform ตอนเทรน ดังนั้นเราจะใช้ transform โดยตรง
    # อย่างไรก็ตาม imputer ถูก fit ด้วยข้อมูลเทรนแล้ว ดังนั้นใช้ transform ได้เลย
    
    # แยก numeric และ categorical
    num_features = ['Recency', 'NumCatalogPurchases', 'NumStorePurchases',
                    'NumWebVisitsMonth', 'Age', 'Total_Promo', 'Teenhome']
    cat_features = ['Education', 'Marital_Status']
    
    # แปลง categorical ด้วย LabelEncoder (ต้องจัดการ Unknown ถ้าไม่เคยเห็น)
    # ใช้ classes_ จาก encoder ที่โหลดมา
    try:
        input_data['Education'] = le_edu.transform(input_data['Education'])
    except ValueError:
        # ถ้าค่าไม่เคยเห็น ให้ใช้ Unknown ที่อยู่ใน classes_ (หรือกำหนดเป็นค่า default)
        input_data['Education'] = le_edu.transform(['Unknown'])[0]  # สมมติ Unknown ถูก encode แล้ว
    
    try:
        input_data['Marital_Status'] = le_mar.transform(input_data['Marital_Status'])
    except ValueError:
        input_data['Marital_Status'] = le_mar.transform(['Unknown'])[0]
    
    # ตัวเลขใช้ imputer (แต่เราไม่ได้ใส่ missing เพราะกรอกครบ)
    input_data[num_features] = num_imp.transform(input_data[num_features])
    
    # ทำนาย
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"🎉 ลูกค้ารายนี้ **มีแนวโน้มตอบรับแคมเปญ** (ความน่าจะเป็น {proba[1]:.2f})")
    else:
        st.error(f"😞 ลูกค้ารายนี้ **ไม่น่าตอบรับแคมเปญ** (ความน่าจะเป็นไม่ตอบรับ {proba[0]:.2f})")
