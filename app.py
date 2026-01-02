import streamlit as st
import pandas as pd
import joblib
import os
import re
import string
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
MODELS_DIR = "models"
REVIEW_FILE = os.path.join(DATASET_DIR, "admin_review.csv")

for folder in [DATASET_DIR, MODELS_DIR]:
    if not os.path.exists(folder): os.makedirs(folder)

# Initialize session state to track which categories need retraining
if 'pending_retrain' not in st.session_state:
    st.session_state.pending_retrain = set()

# --- CORE FUNCTIONS ---

def clean_input(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

def retrain_model(category):
    filepath = os.path.join(DATASET_DIR, f"{category.lower()}.csv")
    if not os.path.exists(filepath): return False
    
    try:
        df = pd.read_csv(filepath).dropna()
        if len(df['label'].unique()) < 2: return False

        tfidf = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1, 3))
        X_tfidf = tfidf.fit_transform(df['text'].values.astype('U'))
        y = df['label']
        
        selector = SelectKBest(chi2, k=min(1200, X_tfidf.shape[1]))
        X_reduced = selector.fit_transform(X_tfidf, y)
        
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        model = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced', probability=True)
        model.fit(X_scaled, y)
        
        model_pack = {'vectorizer': tfidf, 'selector': selector, 'scaler': scaler, 'model': model}
        joblib.dump(model_pack, f"{MODELS_DIR}/{category.lower()}_svm.pkl")
        return True
    except Exception as e:
        st.error(f"Error training {category}: {e}")
        return False

# --- UI LAYOUT ---
st.set_page_config(page_title="CIVerify", layout="wide")
tab1, tab2 = st.tabs(["🔍 News Verifier", "🔐 Admin Review Queue"])

categories = ['business', 'entertainment', 'health', 'politics', 'science', 'sports', 'technology']

# --- TAB 1: PREDICTION ---
with tab1:
    st.title("🛡️ CIVerify: Fake News Detector")
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_cat = st.selectbox("Category", [c.capitalize() for c in categories])
    with col2:
        user_text = st.text_area("Enter content to verify:", height=150)

    if st.button("Analyze News Content", use_container_width=True):
        model_path = os.path.join(MODELS_DIR, f"{selected_cat.lower()}_svm.pkl")
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            # Transformation and Prediction...
            clean = clean_input(user_text)
            vec = data['vectorizer'].transform([clean])
            sel = data['selector'].transform(vec)
            final = data['scaler'].transform(sel)
            prediction = data['model'].predict(final)[0]
            probs = data['model'].predict_proba(final)[0]
            confidence = probs[prediction] * 100
            label = "REAL" if prediction == 1 else "FAKE"

            if confidence < 75:
                st.warning(f"⚠️ Low Confidence ({confidence:.1f}%). Sent to Review.")
                new_review = pd.DataFrame([[user_text, selected_cat.lower(), confidence]], columns=['text', 'category', 'conf'])
                new_review.to_csv(REVIEW_FILE, mode='a', header=not os.path.exists(REVIEW_FILE), index=False)
            
            if label == "REAL": st.success(f"### RESULT: REAL NEWS ({confidence:.2f}%)")
            else: st.error(f"### RESULT: FAKE NEWS ({confidence:.2f}%)")
        else:
            st.info("Model not found. Please train in Admin tab.")

# --- TAB 2: ADMIN PANEL ---
with tab2:
    st.header("Human-in-the-Loop Review")
    pwd = st.sidebar.text_input("Admin Access", type="password")
    
    if pwd == "admin123":
        # --- DYNAMIC RETRAIN SECTION ---
        st.subheader("Model Management")
        if st.session_state.pending_retrain:
            pending_list = ", ".join([c.capitalize() for c in st.session_state.pending_retrain])
            st.warning(f"Pending updates for: **{pending_list}**")
            
            if st.button(f"🔄 RETRAIN {len(st.session_state.pending_retrain)} UPDATED MODELS", type="primary", use_container_width=True):
                with st.spinner("Training only updated categories..."):
                    # Create a copy to iterate because we'll modify the set
                    for c in list(st.session_state.pending_retrain):
                        if retrain_model(c):
                            st.session_state.pending_retrain.remove(c)
                    st.success("Selected models are now up to date!")
                    st.rerun()
        else:
            st.info("All models are currently synchronized with the latest data.")
        
        st.divider()

        # --- REVIEW QUEUE ---
        st.subheader("Verification Queue")
        if os.path.exists(REVIEW_FILE):
            rdf = pd.read_csv(REVIEW_FILE)
            if not rdf.empty:
                curr = rdf.iloc[0]
                with st.container(border=True):
                    st.write(f"Category: **{curr['category'].upper()}**")
                    st.text(curr['text'])
                    
                    c1, c2, c3 = st.columns(3)
                    
                    if c1.button("✅ Confirm REAL", use_container_width=True):
                        cat_path = os.path.join(DATASET_DIR, f"{curr['category']}.csv")
                        pd.DataFrame([[curr['text'], curr['category'], 1]], columns=['text', 'category', 'label']).to_csv(cat_path, mode='a', header=not os.path.exists(cat_path), index=False)
                        
                        # MARK FOR RETRAINING
                        st.session_state.pending_retrain.add(curr['category'])
                        
                        rdf.drop(0).to_csv(REVIEW_FILE, index=False)
                        st.rerun()

                    if c2.button("❌ Confirm FAKE", use_container_width=True):
                        cat_path = os.path.join(DATASET_DIR, f"{curr['category']}.csv")
                        pd.DataFrame([[curr['text'], curr['category'], 0]], columns=['text', 'category', 'label']).to_csv(cat_path, mode='a', header=not os.path.exists(cat_path), index=False)
                        
                        # MARK FOR RETRAINING
                        st.session_state.pending_retrain.add(curr['category'])
                        
                        rdf.drop(0).to_csv(REVIEW_FILE, index=False)
                        st.rerun()
                        
                    if c3.button("🗑️ Discard", use_container_width=True):
                        rdf.drop(0).to_csv(REVIEW_FILE, index=False)
                        st.rerun()
            else: st.success("Queue empty.")
        else: st.info("No items for review.")