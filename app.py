import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

# -------------------- การตั้งค่าหน้าเว็บ --------------------
st.set_page_config(page_title="Marketing Response Prediction", page_icon="📊", layout="wide")

# -------------------- โหลดโมเดลและ LabelEncoder --------------------
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('model_deployment.pkl', 'rb'))
        le_edu = pickle.load(open('encoder_education.pkl', 'rb'))
        le_mar = pickle.load(open('encoder_marital.pkl', 'rb'))
        # ไม่โหลด imputer แล้ว
        return model, le_edu, le_mar
    except FileNotFoundError as e:
        st.error(f"⚠️ ไม่พบไฟล์: {e}")
        st.stop()

model, le_edu, le_mar = load_assets()

# -------------------- ฟังก์ชัน preprocessing (ไม่ใช้ imputer) --------------------
def preprocess_input(df):
    df = df.copy()
    
    required_features = ['Education', 'Marital_Status', 'Teenhome', 'Recency',
                         'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                         'Age', 'Total_Promo']
    
    # ตรวจสอบคอลัมน์ที่ขาด
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        st.error(f"❌ ข้อมูลขาดคอลัมน์: {missing_cols}")
        st.info(f"คอลัมน์ที่มีในไฟล์: {list(df.columns)}")
        return None
    
    # กำหนด default values สำหรับ missing values (ถ้ามี NaN)
    # ใช้ค่าที่ reasonable เช่น median โดยประมาณ (หรือ 0)
    df['Teenhome'] = df['Teenhome'].fillna(0).astype(int)
    df['Recency'] = df['Recency'].fillna(0).astype(int)
    df['NumCatalogPurchases'] = df['NumCatalogPurchases'].fillna(0).astype(int)
    df['NumStorePurchases'] = df['NumStorePurchases'].fillna(0).astype(int)
    df['NumWebVisitsMonth'] = df['NumWebVisitsMonth'].fillna(0).astype(int)
    df['Age'] = df['Age'].fillna(35).astype(int)
    df['Total_Promo'] = df['Total_Promo'].fillna(0).astype(int)
    
    # จัดการ categorical: ถ้า NaN ให้ใช้ 'Unknown'
    df['Education'] = df['Education'].fillna('Unknown').astype(str)
    df['Marital_Status'] = df['Marital_Status'].fillna('Unknown').astype(str)
    
    # Encode ด้วย LabelEncoder (ต้อง map ค่าที่ไม่รู้จักให้เป็น 0 หรือค่าที่เหมาะสม)
    # เนื่องจาก le_edu มี classes ที่กำหนดไว้ ถ้ามีค่าใหม่ที่ไม่เคยเห็นจะ error
    # ดังนั้นเราต้อง map ค่า unseen ไปเป็น 'Unknown' ก่อน
    known_edu = set(le_edu.classes_)
    df['Education'] = df['Education'].apply(lambda x: x if x in known_edu else 'Unknown')
    
    known_mar = set(le_mar.classes_)
    df['Marital_Status'] = df['Marital_Status'].apply(lambda x: x if x in known_mar else 'Unknown')
    
    # Transform
    df['Education'] = le_edu.transform(df['Education'])
    df['Marital_Status'] = le_mar.transform(df['Marital_Status'])
    
    return df[required_features]
