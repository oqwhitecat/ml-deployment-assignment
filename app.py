import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# -------------------- การตั้งค่าหน้าเว็บ --------------------
st.set_page_config(
    page_title="Marketing Response Prediction",
    page_icon="📊",
    layout="wide"
)

# -------------------- โหลดโมเดลและ preprocessing tools --------------------
@st.cache_resource
def load_assets():
    """โหลดโมเดลและตัวแปลงต่าง ๆ จากไฟล์ .pkl"""
    try:
        model = pickle.load(open('model_deployment.pkl', 'rb'))
        le_edu = pickle.load(open('encoder_education.pkl', 'rb'))
        le_mar = pickle.load(open('encoder_marital.pkl', 'rb'))
        num_imputer = pickle.load(open('num_imputer.pkl', 'rb'))
        cat_imputer = pickle.load(open('cat_imputer.pkl', 'rb'))
        return model, le_edu, le_mar, num_imputer, cat_imputer
    except FileNotFoundError as e:
        st.error(f"⚠️ ไม่พบไฟล์: {e}")
        st.info("กรุณาตรวจสอบว่าไฟล์ .pkl ทั้ง 5 อยู่ใน directory เดียวกับแอป")
        st.stop()

model, le_edu, le_mar, num_imputer, cat_imputer = load_assets()

# -------------------- ฟังก์ชันสำหรับ preprocessing ข้อมูลใหม่ --------------------
def preprocess_input(df):
    """Preprocess ข้อมูลที่รับเข้ามาให้ตรงกับตอนเทรนโมเดล"""
    df = df.copy()
    
    # กำหนดค่าเริ่มต้นสำหรับฟีเจอร์ที่อาจหายไป
    if 'Teenhome' not in df.columns:
        df['Teenhome'] = 0
    if 'Recency' not in df.columns:
        df['Recency'] = 0
    if 'NumCatalogPurchases' not in df.columns:
        df['NumCatalogPurchases'] = 0
    if 'NumStorePurchases' not in df.columns:
        df['NumStorePurchases'] = 0
    if 'Age' not in df.columns:
        df['Age'] = 35
    if 'Total_Promo' not in df.columns:
        df['Total_Promo'] = 0
    
    # ตรวจสอบฟีเจอร์ที่ต้องมีทั้งหมด
    required_features = ['Education', 'Marital_Status', 'Teenhome', 'Recency',
                        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                        'Age', 'Total_Promo']
    
    for col in required_features:
        if col not in df.columns:
            st.error(f"❌ ข้อมูลขาดคอลัมน์: {col}")
            return None
    
    # แยกประเภทฟีเจอร์
    num_features = ['Recency', 'NumCatalogPurchases', 'NumStorePurchases',
                    'NumWebVisitsMonth', 'Age', 'Total_Promo', 'Teenhome']
    cat_features = ['Education', 'Marital_Status']
    
    # จัดการ Missing Values (ใช้ imputer ที่บันทึกไว้)
    df[num_features] = num_imputer.transform(df[num_features])
    df[cat_features] = cat_imputer.transform(df[cat_features])
    
    # Encode ข้อความ
    df['Education'] = le_edu.transform(df['Education'].astype(str))
    df['Marital_Status'] = le_mar.transform(df['Marital_Status'].astype(str))
    
    return df[required_features]

# -------------------- ส่วนหัวของแอป --------------------
st.title("🎯 Marketing Campaign Response Prediction")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h4>ระบบทำนายการตอบรับแคมเปญการตลาดด้วย AI</h4>
        <p>เลือกแท็บด้านล่างเพื่อเริ่มต้น: ทดสอบรายบุคคล หรือ อัปโหลดไฟล์ CSV</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- สร้าง Tabs --------------------
tab1, tab2, tab3 = st.tabs(["🔍 ทดสอบรายบุคคล", "📁 อัปโหลด CSV", "📊 ดูข้อมูลโมเดล"])

# ==================== แท็บ 1: ทดสอบรายบุคคล ====================
with tab1:
    st.subheader("🔍 กรอกข้อมูลลูกค้าเพื่อทำนาย")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**ข้อมูลทั่วไป**")
        education_options = ['Unknown', '2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD']
        education = st.selectbox("ระดับการศึกษา", education_options, index=2)
        
        marital_options = ['Unknown', 'Absurd', 'Alone', 'Divorced', 'Married', 'Single', 'Together', 'Widow', 'YOLO']
        marital = st.selectbox("สถานภาพ", marital_options, index=4)
        
        age = st.number_input("อายุ", min_value=18, max_value=100, value=35, step=1)
    
    with col2:
        st.markdown("**พฤติกรรมการซื้อ**")
        recency = st.number_input("จำนวนวันตั้งแต่ซื้อล่าสุด", min_value=0, max_value=100, value=10, step=1)
        teenhome = st.radio("มีลูกวัยรุ่นในบ้านหรือไม่?", [0, 1], format_func=lambda x: "มี (1)" if x == 1 else "ไม่มี (0)")
        num_catalog = st.number_input("ซื้อผ่าน Catalog (ครั้ง)", min_value=0, max_value=50, value=2, step=1)
    
    with col3:
        st.markdown("**ช่องทางดิจิทัล**")
        num_store = st.number_input("ซื้อหน้าร้าน (ครั้ง)", min_value=0, max_value=50, value=3, step=1)
        num_web = st.number_input("เข้าชมเว็บไซต์/เดือน", min_value=0, max_value=50, value=5, step=1)
        total_promo = st.number_input("จำนวนแคมเปญที่เคยตอบรับ", min_value=0, max_value=10, value=1, step=1)
    
    if st.button("🚀 เริ่มทำนาย", use_container_width=True):
        # สร้าง DataFrame จากข้อมูลที่กรอก
        input_data = pd.DataFrame({
            'Education': [education],
            'Marital_Status': [marital],
            'Teenhome': [teenhome],
            'Recency': [recency],
            'NumCatalogPurchases': [num_catalog],
            'NumStorePurchases': [num_store],
            'NumWebVisitsMonth': [num_web],
            'Age': [age],
            'Total_Promo': [total_promo]
        })
        
        # Preprocess
        processed_data = preprocess_input(input_data)
        
        if processed_data is not None:
            # ทำนาย
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            # แสดงผล
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.success(f"### ✅ ผลการทำนาย: **ตอบรับแคมเปญ**")
                    st.metric("โอกาสตอบรับ", f"{probability[1]:.2%}")
                else:
                    st.warning(f"### ❌ ผลการทำนาย: **ไม่ตอบรับแคมเปญ**")
                    st.metric("โอกาสไม่ตอบรับ", f"{probability[0]:.2%}")
            
            with col_res2:
                st.markdown("**คำแนะนำ:**")
                if prediction == 1:
                    st.info("🎯 ลูกค้ามีแนวโน้มสูงที่จะตอบรับ แนะนำให้ส่งแคมเปญพิเศษหรือส่วนลด")
                else:
                    st.info("📚 ลูกค้ายังไม่พร้อมตอบรับ แนะนำให้ส่งเนื้อหาสร้างการรับรู้ก่อน")

# ==================== แท็บ 2: อัปโหลด CSV ====================
with tab2:
    st.subheader("📁 อัปโหลดไฟล์ CSV เพื่อทำนายทีละหลายรายการ")
    
    st.markdown("""
        **รูปแบบไฟล์ที่ต้องการ:**
        - ต้องมีคอลัมน์ตามนี้: `Education`, `Marital_Status`, `Teenhome`, `Recency`,
          `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `Age`, `Total_Promo`
        - ไฟล์ต้องเป็น CSV และใช้คอมม่า (`,`) เป็นตัวคั่น
    """)
    
    uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ อัปโหลดสำเร็จ! พบ {len(df_input)} แถว")
            
            # แสดงตัวอย่างข้อมูล
            with st.expander("🔍 ดูตัวอย่างข้อมูล"):
                st.dataframe(df_input.head())
            
            if st.button("🚀 ทำนายทั้งหมด", use_container_width=True):
                with st.spinner("กำลังประมวลผล..."):
                    # Preprocess ทีละ batch
                    processed_data = preprocess_input(df_input)
                    
                    if processed_data is not None:
                        # ทำนาย
                        predictions = model.predict(processed_data)
                        probabilities = model.predict_proba(processed_data)
                        
                        # เพิ่มผลลัพธ์ลงใน DataFrame
                        df_result = df_input.copy()
                        df_result['Prediction'] = predictions
                        df_result['Prediction_Label'] = df_result['Prediction'].map({0: 'ไม่ตอบรับ', 1: 'ตอบรับ'})
                        df_result['Probability_0'] = probabilities[:, 0]
                        df_result['Probability_1'] = probabilities[:, 1]
                        
                        st.success("✅ ทำนายเสร็จสิ้น!")
                        
                        # แสดงสรุป
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("จำนวนทั้งหมด", len(df_result))
                        with col2:
                            st.metric("ตอบรับ", (df_result['Prediction'] == 1).sum())
                        with col3:
                            st.metric("ไม่ตอบรับ", (df_result['Prediction'] == 0).sum())
                        
                        # แสดงตารางผลลัพธ์
                        st.subheader("📊 ผลลัพธ์การทำนาย")
                        st.dataframe(df_result)
                        
                        # ให้ดาวน์โหลด
                        csv = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 ดาวน์โหลดผลลัพธ์ (CSV)",
                            data=csv,
                            file_name='prediction_results.csv',
                            mime='text/csv'
                        )
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {e}")

# ==================== แท็บ 3: ดูข้อมูลโมเดล ====================
with tab3:
    st.subheader("📊 ข้อมูลเกี่ยวกับโมเดล")
    
    st.markdown("""
        ### ภาพรวมโมเดล
        - **อัลกอริทึม:** Random Forest (ปรับแต่งด้วย GridSearchCV)
        - **จำนวนฟีเจอร์:** 9 ตัว
        - **ข้อมูลที่ใช้เทรน:** 12,240 รายการ (จาก 2 datasets)
        - **การจัดการข้อมูลไม่สมดุล:** ใช้ SMOTE
        
        ### ฟีเจอร์ที่ใช้
        1. Education (ระดับการศึกษา)
        2. Marital_Status (สถานภาพสมรส)
        3. Teenhome (มีลูกวัยรุ่นในบ้าน)
        4. Recency (จำนวนวันตั้งแต่ซื้อล่าสุด)
        5. NumCatalogPurchases (จำนวนการซื้อผ่าน Catalog)
        6. NumStorePurchases (จำนวนการซื้อหน้าร้าน)
        7. NumWebVisitsMonth (จำนวนครั้งที่เข้าเว็บ/เดือน)
        8. Age (อายุ)
        9. Total_Promo (จำนวนแคมเปญที่เคยตอบรับ)
    """)
    
    # แสดง Feature Importance (ถ้าโมเดลมี)
    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 ความสำคัญของฟีเจอร์ (Feature Importance)")
        
        feature_names = ['Education', 'Marital_Status', 'Teenhome', 'Recency',
                         'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                         'Age', 'Total_Promo']
        
        importances = model.feature_importances_
        
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(df_importance.set_index('Feature'))
        st.dataframe(df_importance)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/machine-learning.png", width=80)
    st.markdown("## เกี่ยวกับระบบ")
    st.info("""
        **Marketing Response Prediction**  
        พัฒนาโดย:  
        - ทัดพงศ์ พงศ์สุวากร (663450039-7)  
        - ณัฐธนาภรณ์ อุ้ยเพชร (663450309-4)  
        
        **แหล่งข้อมูล:**  
        - Customer Personality Analysis (Kaggle)  
        - Marketing and Product Performance Dataset (Kaggle)  
        
        **เวอร์ชัน:** 1.0  
        **วันที่:** มีนาคม 2026
    """)
    
    st.markdown("---")
    st.markdown("### วิธีการใช้งาน")
    st.markdown("""
        1. เลือกแท็บ **ทดสอบรายบุคคล** เพื่อทดสอบทีละคน  
        2. หรือเลือกแท็บ **อัปโหลด CSV** เพื่อทำนายทีละหลายรายการ  
        3. ดูรายละเอียดโมเดลได้ในแท็บ **ดูข้อมูลโมเดล**
    """)
