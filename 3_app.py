import streamlit as st
import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. 網頁基本設定與客製化 CSS (讓外觀更現代)
# ==========================================
st.set_page_config(
    page_title="IMDb Sentiment Engine", 
    page_icon="🎬", 
    layout="wide", # 改成寬版，看起來更像專業儀表板
    initial_sidebar_state="expanded"
)

# 客製化 CSS，讓按鈕和輸入框有圓角，字體更大氣
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 1.1rem !important;
        border-radius: 10px !important;
    }
    .stButton>button {
        border-radius: 8px !important;
        font-weight: bold !important;
        height: 50px !important;
        font-size: 1.1rem !important;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E1E1E;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 側邊欄 (Sidebar)：學術火力展示區
# ==========================================
with st.sidebar:
    # 放一個 IMDb 的 Logo 增加質感
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=150)
    st.markdown("### ⚙️ Model Specifications")
    st.markdown("---")
    st.markdown("**Architecture:** BERT-Tiny")
    st.markdown("**Technique:** Knowledge Distillation")
    st.markdown("**Teacher:** `bert-base-uncased`")
    st.markdown("**Training Data:** 50,000 Samples")
    # 直接把你的戰績寫在網頁上！
    st.markdown("**Test Accuracy:** `83.45%`")
    st.markdown("---")
    st.info("💡 **Note:** This model is optimized for extreme low-latency inference while maintaining high accuracy via KL Divergence distillation.")

# ==========================================
# 3. 載入模型 (利用 st.cache_resource 加速)
# ==========================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    model_dir = "./results_distilled_tiny"
    
    if os.path.exists(model_dir):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            model_path = os.path.join(model_dir, latest_checkpoint)
        else:
            return None, None, None
    else:
        return None, None, None

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval() 
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ==========================================
# 4. 主畫面 (Main UI Layout)
# ==========================================
st.markdown('<p class="main-header">🎬 IMDb Sentiment Analysis Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Knowledge-Distilled BERT-Tiny</p>', unsafe_allow_html=True)

# 輸入區
st.markdown("#### Enter Movie Review:")
# 預設一段很棒的英文評論，讓你報告時可以直接點按鈕
default_text = "This movie was an absolute masterpiece! The cinematography was stunning, and the acting was brilliant from start to finish."
user_input = st.text_area("Review Text", value=default_text, height=150, label_visibility="collapsed")

# 執行按鈕
if st.button("🚀 Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review to analyze.")
    elif model is None:
        st.error("❌ Failed to load the model checkpoint. Please verify your training directory.")
    else:
        with st.spinner("Processing through neural network..."):
            
            # --- 核心推論 ---
            start_time = time.time()
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
                
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                confidence = probabilities[0][prediction].item() * 100
                
            # 🌟 把秒數換算成「毫秒 (ms)」，看起來更極速！
            inference_time = (time.time() - start_time) * 1000 

            # --- 顯示結果 ---
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("✅ **Positive**")
                else:
                    st.error("🛑 **Negative**")
                    
            with col2:
                st.metric(label="🎯 Confidence Score", value=f"{confidence:.2f}%")
                
            with col3:
                st.metric(label="⚡ Inference Latency", value=f"{inference_time:.1f} ms")

            # 視覺化信心條
            st.markdown("#### Confidence Distribution")
            if prediction == 1:
                st.progress(confidence / 100, text="Positive Probability")
            else:
                st.progress(confidence / 100, text="Negative Probability")

            # 🌟 學術加分項：隱藏的原始數據面板
            with st.expander("🔍 View Raw Neural Network Output (Logits)"):
                st.json({
                    "Negative Class (0) Logit": float(logits[0][0]),
                    "Positive Class (1) Logit": float(logits[0][1]),
                    "Device Used": str(device).upper()
                })