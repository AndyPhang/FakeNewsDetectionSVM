import streamlit as st
import pandas as pd
import joblib
import os
import io
import re
import string
from github import Github, Auth
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Cloud News Detector", layout="wide")

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_NAME = st.secrets["REPO_NAME"]
except Exception:
    st.error("Missing GitHub Secrets! Please add GITHUB_TOKEN and REPO_NAME to Streamlit Cloud Secrets.")
    st.stop()

# --- 2. GITHUB HELPERS ---
def get_github_file(path):
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(REPO_NAME)
    
    file_content = repo.get_contents(path)
    content_bytes = file_content.decoded_content
    
    if content_bytes is None or len(content_bytes) == 0:
        raise ValueError(f"The file at {path} is empty or contains no headers.")
        
    return file_content, content_bytes

def update_github_file(path, new_content_df, commit_msg):
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(REPO_NAME)
    file_content = repo.get_contents(path)
    
    # Force UTF-8 encoding when saving back to GitHub
    csv_string = new_content_df.to_csv(index=False, encoding='utf-8')
    repo.update_file(path, commit_msg, csv_string, file_content.sha)

# --- 3. CORE FUNCTIONS ---
def clean_input(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

def retrain_and_deploy(category):
    with st.spinner(f"Retraining {category}..."):
        try:
            _, content_bytes = get_github_file(f"dataset/{category.lower()}.csv")
            content_str = content_bytes.decode('utf-8')
            df = pd.read_csv(io.StringIO(content_str)).dropna()
            
            if df.empty:
                st.error(f"Dataset for {category} is empty.")
                return False
            
            tfidf = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1, 3))
            X_tfidf = tfidf.fit_transform(df['text'].values.astype('U')) 
            y = df['label']
            
            selector = SelectKBest(chi2, k=min(1200, X_tfidf.shape[1]))
            X_reduced = selector.fit_transform(X_tfidf, y)
            scaler = MaxAbsScaler()
            X_scaled = scaler.fit_transform(X_reduced)
            
            model = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced', probability=True)
            model.fit(X_scaled, y)
            
            if not os.path.exists("models"): os.makedirs("models")
            model_pack = {'vectorizer': tfidf, 'selector': selector, 'scaler': scaler, 'model': model}
            joblib.dump(model_pack, f"models/{category.lower()}_svm.pkl")
            return True
            
        except Exception as e:
            st.error(f"Retrain Error for {category}: {e}")
            return False

# --- 4. APP UI ---
tab1, tab2 = st.tabs(["🔍 Predict News", "🔐 Admin Review"])
categories = ['business', 'entertainment', 'health', 'politics', 'science', 'sports', 'technology']

with tab1:
    st.title("Collective Intelligence Verifier")
    cat_select = st.selectbox("Category", [c.capitalize() for c in categories], key="sel_cat")
    user_text = st.text_area("Paste text here:", key="txt_input")

    if st.button("Analyze", key="btn_predict"):
        model_path = f"models/{cat_select.lower()}_svm.pkl"
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            clean = clean_input(user_text)
            
            vec = data['vectorizer'].transform([clean])
            sel = data['selector'].transform(vec)
            final = data['scaler'].transform(sel)
            
            prediction = data['model'].predict(final)[0]
            probs = data['model'].predict_proba(final)[0]
            confidence = probs[prediction] * 100
            label = "REAL" if prediction == 1 else "FAKE"

            if confidence < 75:
                st.warning(f"Low confidence ({confidence:.1f}%). Sending to Admin Queue.")
                try:
                    _, content_bytes = get_github_file("dataset/admin_review.csv")
                    rdf = pd.read_csv(io.StringIO(content_bytes.decode('utf-8')))
                    new_entry = pd.DataFrame([[user_text, cat_select.lower(), confidence]], columns=['text', 'category', 'conf'])
                    update_github_file("dataset/admin_review.csv", pd.concat([rdf, new_entry], ignore_index=True), "Added to queue")
                except Exception as e:
                    st.error(f"Queue Error: {e}")
            
            st.write(f"### Result: {label} ({confidence:.2f}%)")
        else:
            st.info("Model not loaded. Use Admin tab to retrain.")

with tab2:
    st.header("Human-in-the-Loop Review")
    admin_pwd = st.sidebar.text_input("Password", type="password", key="admin_pwd")
    
    if admin_pwd == "admin123":
        try:
            # 1. Fetch Review Queue from GitHub
            file_obj, content_bytes = get_github_file("dataset/admin_review.csv")
            review_df = pd.read_csv(io.StringIO(content_bytes.decode('utf-8')))
            
            if not review_df.empty:
                st.write(f"Items Pending: {len(review_df)}")
                curr = review_df.iloc[0]
                st.info(f"Target Category: {curr['category']}")
                st.code(curr['text'])
                
                c1, c2 = st.columns(2)
                
                # --- ACTION: CONFIRM REAL ---
                if c1.button("✅ Label REAL", key="btn_real"):
                    cat_path = f"dataset/{curr['category']}.csv"
                    _, cat_bytes = get_github_file(cat_path)
                    cat_df = pd.read_csv(io.StringIO(cat_bytes.decode('utf-8')))
                    
                    new_row = pd.DataFrame([[curr['text'], curr['category'], 1]], columns=['text', 'category', 'label'])
                    update_github_file(cat_path, pd.concat([cat_df, new_row], ignore_index=True), "Admin: Verified REAL")
                    update_github_file("dataset/admin_review.csv", review_df.drop(0), "Resolved item")
                    
                    st.success(f"Modified {cat_path}!")
                    st.rerun()

                # --- ACTION: CONFIRM FAKE ---
                if c2.button("❌ Label FAKE", key="btn_fake"):
                    cat_path = f"dataset/{curr['category']}.csv"
                    _, cat_bytes = get_github_file(cat_path)
                    cat_df = pd.read_csv(io.StringIO(cat_bytes.decode('utf-8')))
                    
                    new_row = pd.DataFrame([[curr['text'], curr['category'], 0]], columns=['text', 'category', 'label'])
                    update_github_file(cat_path, pd.concat([cat_df, new_row], ignore_index=True), "Admin: Verified FAKE")
                    update_github_file("dataset/admin_review.csv", review_df.drop(0), "Resolved item")
                    
                    st.success(f"Modified {cat_path}!")
                    st.rerun()
            else:
                st.info("Queue is empty.")
                
            st.divider()
            if st.button("🚀 RETRAIN ALL FROM GITHUB", key="btn_retrain", type="primary"):
                for c in categories:
                    retrain_and_deploy(c)
                st.success("All models updated in app memory!")
                
        except Exception as e:
            st.error(f"GitHub Admin Error: {e}")
    else:
        st.write("Enter admin password to access.")