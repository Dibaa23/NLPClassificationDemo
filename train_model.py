#!/usr/bin/env python3
"""
Aircraft Maintenance Log Classification - Model Training Script

This script runs the complete training pipeline without requiring Jupyter notebook.
It generates synthetic data, trains the model, and saves all necessary files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

def generate_synthetic_data():
    """Generate synthetic maintenance log data"""
    print("Generating synthetic maintenance log data...")
    
    # Define categories and their associated keywords/phrases
    categories = {
        'Hydraulic Issue': [
            'hydraulic pressure low', 'hydraulic fluid leak', 'hydraulic pump failure',
            'hydraulic system malfunction', 'hydraulic reservoir empty', 'hydraulic line rupture',
            'hydraulic actuator stuck', 'hydraulic pressure gauge reading zero',
            'hydraulic fluid contamination', 'hydraulic filter clogged',
            'hydraulic pressure relief valve stuck', 'hydraulic accumulator failure',
            'hydraulic system overheating', 'hydraulic fluid level low',
            'hydraulic pump making unusual noise', 'hydraulic pressure fluctuating',
            'hydraulic system slow response', 'hydraulic fluid dark colored',
            'hydraulic pressure drop during operation'
        ],
        'Electrical Issue': [
            'electrical system failure', 'avionics malfunction', 'wiring short circuit',
            'battery voltage low', 'generator not charging', 'electrical panel fault',
            'circuit breaker tripped', 'electrical connection loose', 'voltage regulator failure',
            'electrical system overload', 'wiring harness damaged', 'electrical ground fault',
            'electrical component overheating', 'electrical system noise', 'voltage drop detected',
            'electrical system intermittent', 'electrical connector corrosion',
            'electrical system power loss', 'electrical component failure',
            'electrical system voltage fluctuation', 'electrical wiring insulation damaged'
        ],
        'Landing Gear Issue': [
            'landing gear not retracting', 'landing gear warning light', 'tire pressure low',
            'landing gear hydraulic failure', 'tire wear excessive', 'landing gear strut leak',
            'landing gear door malfunction', 'tire puncture detected', 'landing gear extension failure',
            'landing gear shock absorber failure', 'tire tread separation', 'landing gear lock failure',
            'landing gear position indicator fault', 'tire sidewall damage', 'landing gear brake failure',
            'landing gear wheel bearing noise', 'tire inflation valve leak', 'landing gear actuator failure',
            'landing gear emergency extension', 'landing gear alignment problem', 'tire temperature high'
        ],
        'Engine Issue': [
            'engine performance degradation', 'fuel system malfunction', 'engine oil pressure low',
            'engine temperature high', 'fuel pump failure', 'engine vibration excessive',
            'engine oil consumption high', 'fuel filter clogged', 'engine power loss',
            'engine oil leak detected', 'fuel pressure low', 'engine exhaust smoke',
            'engine starter failure', 'fuel injector malfunction', 'engine oil temperature high',
            'engine compression low', 'fuel system contamination', 'engine bearing noise',
            'engine fuel flow irregular', 'engine oil filter clogged', 'engine ignition system fault'
        ]
    }

    # Additional context phrases to make logs more realistic
    context_phrases = [
        'during preflight inspection', 'after landing', 'during takeoff', 'in flight',
        'during maintenance check', 'before departure', 'after engine start', 'during taxi',
        'at cruise altitude', 'during approach', 'on ground', 'during climb',
        'after refueling', 'during descent', 'before engine shutdown'
    ]

    # Aircraft identifiers
    aircraft_ids = [
        'Aircraft A320-001', 'Aircraft B737-002', 'Aircraft A380-003', 'Aircraft B787-004',
        'Aircraft A350-005', 'Aircraft B777-006', 'Aircraft A330-007', 'Aircraft B747-008'
    ]

    # Generate synthetic data
    data = []
    np.random.seed(42)  # For reproducibility

    for category, phrases in categories.items():
        for i in range(50):  # 50 samples per category
            # Select random phrase and context
            main_phrase = np.random.choice(phrases)
            context = np.random.choice(context_phrases)
            aircraft = np.random.choice(aircraft_ids)
            
            # Create realistic log entry
            log_entry = f"{aircraft}: {main_phrase} {context}. "
            
            # Add some variation with additional details
            if np.random.random() > 0.5:
                severity = np.random.choice(['Minor', 'Moderate', 'Major'])
                log_entry += f"Severity: {severity}. "
            
            if np.random.random() > 0.7:
                action = np.random.choice([
                    'Maintenance crew notified.', 'Pilot reported issue.',
                    'Technician inspection required.', 'Immediate attention needed.'
                ])
                log_entry += action
            
            data.append({
                'aircraft_id': aircraft,
                'maintenance_log': log_entry.strip(),
                'category': category
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} maintenance log entries")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    
    return df

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def train_and_evaluate_model(df):
    """Train and evaluate the classification model"""
    print("Preprocessing text data...")
    
    # Apply preprocessing
    df['cleaned_text'] = df['maintenance_log'].apply(preprocess_text)
    
    # Split the data
    X = df['cleaned_text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # TF-IDF Vectorization
    print("Vectorizing text data with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # Fit and transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")

    # Train Logistic Regression
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)

    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)

    # Make predictions
    lr_pred = lr_model.predict(X_test_tfidf)
    rf_pred = rf_model.predict(X_test_tfidf)

    # Evaluate models
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }

    predictions = {
        'Logistic Regression': lr_pred,
        'Random Forest': rf_pred
    }

    print("\nModel Performance Comparison:")
    print("=" * 50)

    best_model = None
    best_accuracy = 0

    for name, pred in predictions.items():
        accuracy = accuracy_score(y_test, pred)
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = models[name]

    return best_model, tfidf_vectorizer, df

def save_models_and_data(model, vectorizer, df):
    """Save the trained model, vectorizer, and dataset"""
    print("\nSaving models and data...")
    
    # Save the best model and vectorizer
    joblib.dump(model, 'maintenance_classifier_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    # Save the dataset for reference
    df.to_csv('maintenance_logs_dataset.csv', index=False)

    print("Models and data saved successfully!")
    print("Files created:")
    print("- maintenance_classifier_model.pkl")
    print("- tfidf_vectorizer.pkl")
    print("- maintenance_logs_dataset.csv")

def test_model(model, vectorizer):
    """Test the saved model with a sample input"""
    print("\nTesting the saved model...")
    
    test_text = "Aircraft A320-001: hydraulic pressure low during preflight inspection. Severity: Moderate."
    cleaned_test = preprocess_text(test_text)
    test_vectorized = vectorizer.transform([cleaned_test])
    prediction = model.predict(test_vectorized)[0]
    probability = np.max(model.predict_proba(test_vectorized))

    print(f"Test Prediction:")
    print(f"Input: {test_text}")
    print(f"Predicted Category: {prediction}")
    print(f"Confidence: {probability:.4f}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Aircraft Maintenance Log Classification - Model Training")
    print("=" * 60)
    
    # Download NLTK data
    download_nltk_data()
    
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Train and evaluate model
    model, vectorizer, df = train_and_evaluate_model(df)
    
    # Save models and data
    save_models_and_data(model, vectorizer, df)
    
    # Test the model
    test_model(model, vectorizer)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("You can now run the Streamlit app with: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 