import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------- การตั้งค่าหน้าเว็บ --------------------
st.set_page_config(
    page_title="ทำนายการตอบรับแคมเปญ",
    page_icon="📊",
    layout="centered"
)

# -------------------- ค่าคงที่จากข้อมูลเดิม (ใช้แทน imputer) --------------------
# ค่า Median จากข้อมูลเดิม (ดูจากโค้ดเทรน)
MEDIAN_AGE = 55
MEDIAN_RECENCY = 49
MEDIAN_CATALOG = 2
MEDIAN_STORE = 5
MEDIAN_TEENHOME = 0
DEFAULT_WEB_VISITS = 3      # ค่ากลาง ๆ
DEFAULT_TOTAL_PROMO = 1     # ค่ากลาง ๆ
CATEGORICAL_DEFAULT = 'Unknown'

# -------------------- โหลดโมเดลและ Encoders --------------------
@st.cache_resource
def load_model_and_encoders():
    """โหลดโมเดล Random Forest และ LabelEncoders"""
    try:
        with open('model_deployment.pkl', 'rb') f:
            model = pickle.load(f)
        with open('encoder_education.pkl', 'rb') f:
            le_edu = pickle.load(f)
        with open('encoder_marital.pkl', 'rb') f:
            le_mar = pickle.load(f)
        return model, le_edu, le_mar
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.stop()

model, le_edu, le_mar = load_model_and_encoders()

# -------------------- ฟังก์ชันเตรียมข้อมูล --------------------
def preprocess_input(df):
    """
    รับ DataFrame ที่มีคอลัมน์ตรงกับที่ user กรอก
    คืนค่า DataFrame ที่ผ่านการ encode และเรียงลำดับคอลัมน์ตามที่โมเดลต้องการ
    """
    # 1. จัดการ Missing Values (ถ้ามี) โดยใช้ค่าคงที่
    df['Age'] = df['Age'].fillna(MEDIAN_AGE)
    df['Recency'] = df['Recency'].fillna(MEDIAN_RECENCY)
    df['NumCatalogPurchases'] = df['NumCatalogPurchases'].fillna(MEDIAN_CATALOG)
    df['NumStorePurchases'] = df['NumStorePurchases'].fillna(MEDIAN_STORE)
    df['Teenhome'] = df['Teenhome'].fillna(MEDIAN_TEENHOME)
    df['NumWebVisitsMonth'] = df['NumWebVisitsMonth'].fillna(DEFAULT_WEB_VISITS)
    df['Total_Promo'] = df['Total_Promo'].fillna(DEFAULT_TOTAL_PROMO)
    df['Education'] = df['Education'].fillna(CATEGORICAL_DEFAULT)
    df['Marital_Status'] = df['Marital_Status'].fillna(CATEGORICAL_DEFAULT)

    # 2. แปลงค่าข้อความด้วย LabelEncoder (จัดการค่าที่ไม่เคยเห็น)
    try:
        df['Education'] = le_edu.transform(df['Education'])
    except ValueError:
        # ถ้าค่าที่กรอกไม่เคยเห็น ให้ใช้ 'Unknown' แทน
        df['Education'] = le_edu.transform([CATEGORICAL_DEFAULT])[0]

    try:
        df['Marital_Status'] = le_mar.transform(df['Marital_Status'])
    except ValueError:
        df['Marital_Status'] = le_mar.transform([CATEGORICAL_DEFAULT])[0]

    # 3. เลือกเฉพาะคอลัมน์ที่โมเดลต้องการ ตามลำดับที่ถูกต้อง
    feature_order = ['Education', 'Marital_Status', 'Teenhome', 'Recency',
                     'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                     'Age', 'Total_Promo']
    return df[feature_order]

# -------------------- ส่วนติดต่อผู้ใช้ --------------------
st.title("📈 ทำนายการตอบรับแคมเปญการตลาด")
st.markdown("กรอกข้อมูลลูกค้าเพื่อประเมินว่า **จะตอบรับแคมเปญ (Response = 1)** หรือไม่")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "ระดับการศึกษา",
            options=['Unknown', 'Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'],
            index=0
        )
        marital_status = st.selectbox(
            "สถานภาพ",
            options=['Unknown', 'Married', 'Together', 'Single', 'Divorced',
                     'Widow', 'Alone', 'Absurd', 'YOLO'],
            index=0
        )
        teenhome = st.number_input("จำนวนวัยรุ่นในบ้าน (Teenhome)", min_value=0, max_value=2, value=0)
        age = st.number_input("อายุ", min_value=18, max_value=100, value=55)

    with col2:
        recency = st.number_input("จำนวนวันตั้งแต่ซื้อครั้งล่าสุด", min_value=0, value=49)
        num_catalog = st.number_input("ซื้อผ่านแคตตาล็อก", min_value=0, value=2)
        num_store = st.number_input("ซื้อที่ร้าน", min_value=0, value=5)
        num_web = st.number_input("เข้าชมเว็บต่อเดือน", min_value=0, value=3)
        total_promo = st.number_input("จำนวนแคมเปญที่ร่วม", min_value=0, value=1)

    submitted = st.form_submit_button("🔮 ทำนาย")

if submitted:
    # สร้าง DataFrame จากข้อมูลที่กรอก
    input_dict = {
        'Education': [education],
        'Marital_Status': [marital_status],
        'Teenhome': [teenhome],
        'Recency': [recency],
        'NumCatalogPurchases': [num_catalog],
        'NumStorePurchases': [num_store],
        'NumWebVisitsMonth': [num_web],
        'Age': [age],
        'Total_Promo': [total_promo]
    }
    input_df = pd.DataFrame(input_dict)

    # ประมวลผล
    processed_df = preprocess_input(input_df)

    # ทำนาย
    prediction = model.predict(processed_df)[0]
    proba = model.predict_proba(processed_df)[0]

    # แสดงผล
    st.markdown("---")
    if prediction == 1:
        st.success(f"### 🎉 ลูกค้ารายนี้ **มีแนวโน้มตอบรับแคมเปญ**")
        st.markdown(f"**ความน่าจะเป็น:** ตอบรับ {proba[1]:.2f} | ไม่ตอบรับ {proba[0]:.2f}")
    else:
        st.error(f"### 😞 ลูกค้ารายนี้ **ไม่น่าตอบรับแคมเปญ**")
        st.markdown(f"**ความน่าจะเป็น:** ไม่ตอบรับ {proba[0]:.2f} | ตอบรับ {proba[1]:.2f}")

    # แสดงข้อมูลที่กรอก (optional)
    with st.expander("ดูข้อมูลที่กรอก"):
        st.dataframe(input_df)

# -------------------- ข้อมูลเพิ่มเติม --------------------
st.sidebar.header("เกี่ยวกับแอป")
st.sidebar.info(
    """
    **โมเดล:** Random Forest (เทรนด้วยข้อมูล Customer Personality Analysis + ข้อมูลเพิ่มเติม)\n
    **Features ที่ใช้:** Education, Marital_Status, Teenhome, Recency, NumCatalogPurchases,
    NumStorePurchases, NumWebVisitsMonth, Age, Total_Promo\n
    **หมายเหตุ:** หากค่าที่เลือกไม่อยู่ในชุดเทรน ระบบจะใช้ 'Unknown' แทน
    """
)
