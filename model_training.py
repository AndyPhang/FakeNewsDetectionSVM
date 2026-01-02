import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score

def train_svm_final_push():
    # Configuration
    dataset_dir = 'dataset'
    models_dir = 'models'
    categories = ['business', 'entertainment', 'health', 'politics', 'science', 'sports', 'technology']
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print(f"{'Category':<15} | {'Best C':<8} | {'Best G':<8} | {'Accuracy'}")
    print("-" * 55)

    for cat in categories:
        # Update path to look inside the dataset folder
        filename = os.path.join(dataset_dir, f"{cat}.csv")
        
        if not os.path.exists(filename):
            print(f"Skipping {cat}: {filename} not found.")
            continue
            
        df = pd.read_csv(filename).dropna()
        
        # 1. Expand N-grams (Using Bigrams as requested by your previous logic)
        tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(2, 2))
        X_tfidf = tfidf.fit_transform(df['text'].values.astype('U'))
        y = df['label']
        
        # 2. Refined Feature Selection
        selector = SelectKBest(chi2, k=min(1200, X_tfidf.shape[1]))
        X_reduced = selector.fit_transform(X_tfidf, y)
        
        # 3. Scaling
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # 4. Grid Search with Probability enabled for Streamlit/Predict confidence
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [1, 0.1, 0.01, 'scale'],
            'kernel': ['rbf']
        }
        
        grid = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid, refit=True, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        # 5. Result
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Save the best version found
        model_pack = {
            'vectorizer': tfidf,
            'selector': selector,
            'scaler': scaler,
            'model': best_model
        }
        model_filename = os.path.join(models_dir, f'{cat}_svm.pkl')
        joblib.dump(model_pack, model_filename)
        
        print(f"{cat.capitalize():<15} | {grid.best_params_['C']:<8} | {grid.best_params_['gamma']:<8} | {acc:<10.2%}")

if __name__ == '__main__':
    train_svm_final_push()