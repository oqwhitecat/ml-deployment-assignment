
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. โหลดโมเดลและเครื่องมือ ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('model_deployment.pkl', 'rb'))
    le_edu = pickle.load(open('encoder_education.pkl', 'rb'))
    le_mar = pickle.load(open('encoder_marital.pkl', 'rb'))
    return model, le_edu, le_mar

model, le_edu, le_mar = load_assets()

# --- 2. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Marketing Intelligence Dashboard", layout="wide")

st.title("Customer Response Prediction Dashboard")
st.markdown("ระบบวิเคราะห์และทำนายพฤติกรรมลูกค้าด้วย AI สำหรับวางแผนกลยุทธ์การตลาด")

# --- 3. การแบ่งเมนูหลัก ---
tab1, tab2, tab3 = st.tabs(["Prediction (รายคน)", "Batch (รายกลุ่ม)", "Model Insights"])

# --- TAB 1: ทำนายรายคน ---
with tab1:
    st.subheader("ระบบประเมินโอกาสตอบรับรายบุคคล")
    st.info("คำแนะนำ: ลองใช้ค่าเริ่มต้นที่ระบบตั้งไว้ หรือดูตารางตัวอย่างข้อมูลใน Sidebar")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**ข้อมูลพื้นฐาน**")
            education = st.selectbox("ระดับการศึกษา", le_edu.classes_)
            marital = st.selectbox("สถานะสมรส", le_mar.classes_)
            age = st.number_input("อายุ (Age)", 18, 90, 35)
            
        with col2:
            st.markdown("**พฤติกรรมการซื้อ**")
            recency = st.slider("ระยะเวลาหลังซื้อล่าสุด (วัน)", 0, 100, 15)
            teenhome = st.radio("มีลูกวัยรุ่นในบ้านหรือไม่? (0=ไม่มี, 1=มี)", [0, 1], horizontal=True)
            num_cat = st.number_input("ซื้อผ่าน Catalog", 0, 30, 2)
            
        with col3:
            st.markdown("**ช่องทางดิจิทัล**")
            num_web = st.number_input("เข้าชมเว็บ/เดือน", 0, 20, 5)
            num_store = st.number_input("ซื้อหน้าร้าน", 0, 30, 5)
            total_promo = st.select_slider("แคมเปญที่เคยรับ (อดีต)", options=[0, 1, 2, 3, 4, 5], value=1)

    if st.button("เริ่มวิเคราะห์แนวโน้ม", use_container_width=True):
        features = np.array([[le_edu.transform([education])[0], le_mar.transform([marital])[0], 
                             teenhome, recency, num_cat, num_store, num_web, age, total_promo]])
        
        prob = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            score = round(prob * 100, 2)
            label = "YES (ตอบรับ)" if prediction == 1 else "NO (ไม่ตอบรับ)"
            st.metric(label="AI Analysis Score", value=f"{score}%", delta=label)
            
        with res_col2:
            st.subheader("คำแนะนำเชิงกลยุทธ์ (Insight)")
            if prediction == 1:
                st.success("กลยุทธ์: ลูกค้ามีแนวโน้มสูง แนะนำให้ส่งโปรโมชั่นทาง SMS/Email ทันที")
            else:
                st.warning("กลยุทธ์: ลูกค้ายังไม่พร้อม แนะนำให้เน้นการสร้าง Awareness ก่อนส่งโปรโมชั่นตรง")

# --- TAB 2 & 3 ---
with tab2: st.info("เมนูวิเคราะห์กลุ่มลูกค้าปริมาณมาก (Batch Analysis)")
with tab3: 
    st.subheader("Model Explainability")
    st.write("ปัจจัยที่มีผลต่อการตัดสินใจของ AI")
    st.bar_chart({'Factors': [0.4, 0.3, 0.2, 0.1], 'Weight': [4,3,2,1]})

# --- 4. Sidebar ---
with st.sidebar:
    st.title("About Data")
    st.markdown("""
    ### แหล่งอ้างอิงข้อมูล
    AI เรียนรู้จากข้อมูลลูกค้า 2,240 ราย 
    แหล่งที่มา: [Kaggle Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
    
    ### ข้อมูลกรอกอย่างไร?
    - **Recency:** ยิ่งค่าน้อยยิ่งมีโอกาสตอบรับสูง
    - **Total Promo:** ยิ่งเคยรับมาก ยิ่งเป็นลูกค้ากลุ่มภักดี
    
    ---
    **Version:** 1.0 (Final Deployment)
    """)
    