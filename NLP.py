import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc, roc_auc_score)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def advanced_clean_text(text):
    """Advanced text cleaning with lemmatization"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() 
                     if word not in stop_words and len(word) > 2])
    return text

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    st.pyplot(fig)

def plot_roc_curve(y_true, y_proba, model_name):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    st.pyplot(fig)

st.set_page_config(page_title="Advanced Spam Detection", layout="wide")
st.title("🚀 Advanced Email Spam Detection System")
st.write("AI-powered spam classifier with multiple ML algorithms and advanced NLP techniques")

# Sidebar
st.sidebar.header("👤 User Details")
name = st.sidebar.text_input("Enter Name", "Guest")
email = st.sidebar.text_input("Enter Email")
st.sidebar.success(f"Welcome {name}!")

st.sidebar.header("⚙️ Model Settings")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.3, 0.2)
max_features = st.sidebar.slider("Max TF-IDF Features", 1000, 5000, 3000, step=500)

# File upload
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check columns
    if 'text' in df.columns and 'spam' in df.columns:
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🎯 Model Training", "📈 Results", "🧪 Test Email"])
        
        with tab1:
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Emails", len(df))
            with col2:
                spam_count = (df['spam'] == 1).sum()
                st.metric("Spam Emails", spam_count)
            with col3:
                ham_count = (df['spam'] == 0).sum()
                st.metric("Ham Emails", ham_count)
            
            st.write("**Label Distribution:**")
            fig, ax = plt.subplots()
            df['spam'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
            ax.set_xticklabels(['Ham (0)', 'Spam (1)'], rotation=0)
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
            st.write("**Sample Data:**")
            st.dataframe(df.head(10))
        
        with tab2:
            st.subheader("Data Preprocessing & Model Training")
            
            progress = st.progress(0)
            
            # Preprocessing
            st.write("🔄 Processing text data...")
            df['cleaned_text'] = df['text'].apply(advanced_clean_text)
            df['label'] = df['spam'].astype(int)
            progress.progress(25)
            
            # Feature Extraction
            st.write("📝 Extracting TF-IDF features...")
            tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
            X = tfidf.fit_transform(df['cleaned_text'])
            y = df['label']
            progress.progress(50)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train Multiple Models
            st.write("🤖 Training multiple models...")
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42)
            }
            
            trained_models = {}
            model_scores = {}
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                model_scores[model_name] = accuracy
            
            progress.progress(100)
            st.success("✅ Training complete!")
            
            # Store in session state for tab3 and tab4
            st.session_state.trained_models = trained_models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.tfidf = tfidf
            st.session_state.model_scores = model_scores
        
        with tab3:
            st.subheader("📊 Model Performance Comparison")
            
            if 'trained_models' in st.session_state:
                models = st.session_state.trained_models
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Model comparison table
                results = []
                for model_name, model in models.items():
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    results.append({
                        'Model': model_name,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1-Score': f1_score(y_test, y_pred),
                        'ROC-AUC': roc_auc_score(y_test, y_proba)
                    })
                
                results_df = pd.DataFrame(results)
                st.write("**Performance Metrics:**")
                st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
                
                # Select best model for visualization
                best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
                best_model = models[best_model_name]
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]
                
                st.write(f"\n**Best Model: {best_model_name}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    plot_confusion_matrix(y_test, y_pred, best_model_name)
                with col2:
                    plot_roc_curve(y_test, y_proba, best_model_name)
                
                # Classification Report
                st.write("**Detailed Classification Report:**")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                st.session_state.best_model = best_model
                st.session_state.best_model_name = best_model_name
        
        with tab4:
            st.subheader("🧪 Test Email on Best Model")
            
            if 'best_model' in st.session_state:
                model = st.session_state.best_model
                tfidf = st.session_state.tfidf
                model_name = st.session_state.best_model_name
                
                st.info(f"Using **{model_name}** model")
                
                user_email = st.text_area("Enter an email message to test:", height=150)
                
                if user_email:
                    cleaned_email = advanced_clean_text(user_email)
                    email_tfidf = tfidf.transform([cleaned_email])
                    
                    prediction = model.predict(email_tfidf)[0]
                    probability = model.predict_proba(email_tfidf)[0]
                    
                    col1, col2 = st.columns(2)
                    
                    if prediction == 1:
                        with col1:
                            st.error(f"🚨 SPAM DETECTED")
                            st.write(f"Confidence: **{probability[1]:.2%}**")
                        with col2:
                            fig, ax = plt.subplots()
                            ax.barh(['Spam', 'Ham'], probability, color=['#e74c3c', '#2ecc71'])
                            ax.set_xlim(0, 1)
                            st.pyplot(fig)
                    else:
                        with col1:
                            st.success(f"✅ LEGITIMATE EMAIL")
                            st.write(f"Confidence: **{probability[0]:.2%}**")
                        with col2:
                            fig, ax = plt.subplots()
                            ax.barh(['Spam', 'Ham'], probability, color=['#e74c3c', '#2ecc71'])
                            ax.set_xlim(0, 1)
                            st.pyplot(fig)
            else:
                st.warning("⚠️ Please train models first in the 'Model Training' tab")
    else:
        st.error("❌ CSV must contain 'text' and 'spam' columns")







