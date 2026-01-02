import streamlit as st
import pandas as pd
import joblib
import os
import io
import re
import string
from github import Github  # pip install PyGithub
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2

# --- STREAMLIT SECRETS (Setup in Streamlit Cloud Dashboard) ---
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO_NAME = st.secrets["REPO_NAME"] # e.g., "username/project-repo"

# --- HELPERS FOR GITHUB PERSISTENCE ---
def get_github_file(path):
    g = Github(st.secrets["GITHUB_TOKEN"])
    repo = g.get_repo(st.secrets["REPO_NAME"])
    file_content = repo.get_contents(path)
    return file_content, file_content.decoded_content

def update_github_file(path, new_content_df, commit_msg):
    g = Github(st.secrets["GITHUB_TOKEN"])
    repo = g.get_repo(st.secrets["REPO_NAME"])
    file_content = repo.get_contents(path)
    # Convert DF to CSV string
    csv_string = new_content_df.to_csv(index=False)
    repo.update_file(path, commit_msg, csv_string, file_content.sha)

# --- CORE FUNCTIONS ---
def clean_input(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

@st.cache_resource # Keeps model in memory for faster performance
def load_category_model(category):
    model_path = f"models/{category.lower()}_svm.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def retrain_and_deploy(category):
    """Retrains using data pulled from GitHub."""
    _, content = get_github_file(f"dataset/{category.lower()}.csv")
    df = pd.read_csv(io.BytesIO(content)).dropna()
    
    # ... (Your training logic remains the same) ...
    tfidf = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1, 3))
    X_tfidf = tfidf.fit_transform(df['text'].values.astype('U'))
    y = df['label']
    selector = SelectKBest(chi2, k=min(1200, X_tfidf.shape[1]))
    X_reduced = selector.fit_transform(X_tfidf, y)
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    model = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced', probability=True)
    model.fit(X_scaled, y)
    
    # Save locally (ephemeral) for immediate use
    model_pack = {'vectorizer': tfidf, 'selector': selector, 'scaler': scaler, 'model': model}
    joblib.dump(model_pack, f"models/{category.lower()}_svm.pkl")
    
    # OPTIONAL: You can also push the .pkl back to GitHub here using update_github_file
    return True

# --- UI LOGIC (Modified for Cloud) ---
st.set_page_config(page_title="Cloud Collective Intelligence")
# ... (Tabs setup) ...

# Inside the Admin "Confirm REAL/FAKE" button:
# Instead of pd.to_csv(cat_path, mode='a'...)
# 1. Get current data from GitHub
file_obj, content = get_github_file(f"dataset/{curr['category']}.csv")
df = pd.read_csv(io.BytesIO(content))

# 2. Add new human-verified row
new_row = pd.DataFrame([[curr['text'], curr['category'], 1]], columns=['text', 'category', 'label'])
updated_df = pd.concat([df, new_row])

# 3. Push back to GitHub (This is the "Persistence")
update_github_file(f"dataset/{curr['category']}.csv", updated_df, "Collective Intelligence Update")
st.toast("Intelligence synced to GitHub!")