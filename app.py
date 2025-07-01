import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Aircraft Maintenance Log Classifier",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid #34495e;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
    }
    
    .metric-card h2 {
        color: #667eea;
        margin: 0;
        font-weight: 700;
    }
    
    .metric-card h3 {
        color: #ecf0f1;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .metric-card h4 {
        color: #ecf0f1;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .metric-card p {
        color: #bdc3c7;
        margin: 0;
    }
    
    .metric-card ul {
        color: #bdc3c7;
    }
    
    .metric-card ol {
        color: #bdc3c7;
    }
    
    /* Category cards */
    .category-card {
        background: linear-gradient(135deg, var(--color) 0%, var(--color-dark) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    
    .category-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .category-card .confidence {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(44, 62, 80, 0.3);
        color: white;
    }
    
    .stFileUploader:hover {
        background: rgba(44, 62, 80, 0.5);
        border-color: #764ba2;
    }
    
    /* Success/Error messages */
    .success-message {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(90deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        border-left: 4px solid var(--priority-color);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .recommendation-box h4 {
        color: var(--priority-color);
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .recommendation-box p {
        color: #bdc3c7;
        margin: 0.5rem 0;
    }
    
    .recommendation-box strong {
        color: #ecf0f1;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Download NLTK data if needed
@st.cache_resource
def download_nltk_data():
    """Download NLTK data with better error handling for cloud deployment"""
    try:
        # Force download of punkt tokenizer
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Verify downloads
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        
        st.success("NLTK data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading NLTK data: {e}")
        # Try alternative approach
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e2:
            st.error(f"Failed to download NLTK data: {e2}")

# Download NLTK data at startup
download_nltk_data()

# Load models and data
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('maintenance_classifier_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please run the Jupyter notebook first to train the model.")
        return None, None

@st.cache_data
def load_dataset():
    """Load the dataset for examples and analysis"""
    try:
        df = pd.read_csv('maintenance_logs_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found! Please run the Jupyter notebook first.")
        return None

def get_recommendations():
    """Get recommendations for each maintenance category"""
    return {
        'Hydraulic Issue': {
            'action': 'Inspect hydraulic pumps and fluid levels.',
            'details': 'Check hydraulic pressure, fluid contamination, pump operation, and system leaks. Verify accumulator pressure and relief valve functionality.',
            'priority': 'High',
            'color': '#FF6B6B',
            'color_dark': '#FF5252',
            'priority_color': '#FF6B6B'
        },
        'Electrical Issue': {
            'action': 'Check auxiliary power systems and circuit breakers.',
            'details': 'Inspect electrical connections, wiring integrity, voltage levels, and circuit breaker status. Test avionics systems and battery condition.',
            'priority': 'High',
            'color': '#4ECDC4',
            'color_dark': '#26A69A',
            'priority_color': '#4ECDC4'
        },
        'Landing Gear Issue': {
            'action': 'Examine landing gear retraction and extension systems.',
            'details': 'Check landing gear hydraulic systems, tire pressure and condition, brake functionality, and position indicators. Inspect strut extension and door operation.',
            'priority': 'Critical',
            'color': '#45B7D1',
            'color_dark': '#1976D2',
            'priority_color': '#FF5722'
        },
        'Engine Issue': {
            'action': 'Inspect engine performance and fuel systems.',
            'details': 'Check engine oil pressure and temperature, fuel flow, compression levels, and vibration. Inspect fuel filters, injectors, and exhaust systems.',
            'priority': 'Critical',
            'color': '#96CEB4',
            'color_dark': '#4CAF50',
            'priority_color': '#FF5722'
        }
    }

def preprocess_text(text):
    """Clean and preprocess text data with fallback for NLTK issues"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        # Try to use NLTK tokenization
        tokens = word_tokenize(text)
        
        # Try to use NLTK stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        except LookupError:
            # Fallback: simple stopwords list
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            tokens = [token for token in tokens if token not in basic_stopwords and len(token) > 2]
            
    except LookupError:
        # Fallback: simple word splitting
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)

def predict_category(text, model, vectorizer):
    """Predict category for given text with confidence"""
    if not text.strip():
        return None, None, None
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Get confidence for predicted category
    confidence = np.max(probabilities)
    
    return prediction, confidence, probabilities

def process_batch_classification(df, model, vectorizer):
    """Process batch classification for CSV upload"""
    results = []
    recommendations = get_recommendations()
    
    for idx, row in df.iterrows():
        log_text = row['Log']
        prediction, confidence, probabilities = predict_category(log_text, model, vectorizer)
        
        if prediction is not None and confidence is not None:
            recommendation = recommendations[prediction]['action']
            results.append({
                'Log': log_text,
                'Predicted Category': prediction,
                'Confidence (%)': f"{confidence * 100:.1f}%",
                'Recommendation': recommendation
            })
        else:
            results.append({
                'Log': log_text,
                'Predicted Category': 'Error',
                'Confidence (%)': '0%',
                'Recommendation': 'Unable to process this entry'
            })
    
    return pd.DataFrame(results)

def create_confidence_chart(categories, probabilities, recommendations):
    """Create an interactive confidence chart using Plotly"""
    colors = [recommendations.get(cat, {}).get('color', '#gray') for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probabilities,
            marker_color=colors,
            text=[f'{prob:.1%}' for prob in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence by Category',
        xaxis_title='Category',
        yaxis_title='Confidence Score',
        yaxis=dict(tickformat='.1%'),
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_category_distribution(category_counts, recommendations):
    """Create an interactive pie chart for category distribution"""
    colors = [recommendations.get(cat, {}).get('color', '#gray') for cat in category_counts.index]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='inside',
            hole=0.4
        )
    ])
    
    fig.update_layout(
        title='Distribution of Predicted Categories',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def main():
    # Header with modern styling
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Aircraft Maintenance Log Classifier</h1>
        <p>AI-Powered Maintenance Issue Classification & Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and data
    model, vectorizer = load_models()
    df = load_dataset()
    
    if model is None or vectorizer is None or df is None:
        st.stop()
    
    # Load recommendations
    recommendations = get_recommendations()
    
    # Sidebar with modern styling
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #667eea; margin-bottom: 1rem;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Single Text Classification", "Batch CSV Processing", "About"]
    )
    
    if page == "Single Text Classification":
        st.markdown("## 🔍 Single Text Classification")
        st.markdown("Enter an aircraft maintenance log entry below to classify it and get actionable recommendations.")
        
        # Text input with modern styling
        user_input = st.text_area(
            "Maintenance Log Entry:",
            placeholder="Enter your maintenance log here...\nExample: Aircraft A320-001: hydraulic pressure low during preflight inspection. Severity: Moderate.",
            height=150
        )
        
        # Classification button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            classify_button = st.button("🚀 Classify Text", type="primary", use_container_width=True)
        
        if classify_button:
            if user_input.strip():
                with st.spinner("🔍 Analyzing text..."):
                    prediction, confidence, probabilities = predict_category(user_input, model, vectorizer)
                    
                    if prediction is not None and confidence is not None:
                        # Display results in modern cards
                        st.markdown("### 📊 Classification Results")
                        
                        # Category card with confidence
                        category_info = recommendations[prediction]
                        confidence_percent = confidence * 100
                        
                        st.markdown(f"""
                        <div class="category-card" style="--color: {category_info['color']}; --color-dark: {category_info['color_dark']};">
                            <h3>{prediction}</h3>
                            <p class="confidence">Confidence: {confidence_percent:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence chart
                        if model is not None and hasattr(model, 'classes_'):
                            categories = model.classes_
                            fig = create_confidence_chart(categories, probabilities, recommendations)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendation section
                        st.markdown("### 💡 Recommended Action")
                        
                        st.markdown(f"""
                        <div class="recommendation-box" style="--priority-color: {category_info['priority_color']};">
                            <h4>Priority: {category_info['priority']}</h4>
                            <p><strong>Action:</strong> {category_info['action']}</p>
                            <p><strong>Details:</strong> {category_info['details']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show preprocessing details
                        with st.expander("🔧 Text Preprocessing Details"):
                            cleaned_text = preprocess_text(user_input)
                            st.write("**Original text:**", user_input)
                            st.write("**Cleaned text:**", cleaned_text)
                            
                    else:
                        st.markdown('<div class="error-message">❌ Please enter valid text for classification.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">⚠️ Please enter some text to classify.</div>', unsafe_allow_html=True)
    
    elif page == "Batch CSV Processing":
        st.markdown("## 📁 Batch CSV Processing")
        st.markdown("Upload a CSV file with maintenance logs for batch classification and analysis.")
        
        # Instructions card
        st.markdown("""
        <div class="metric-card">
            <h4>📋 Instructions</h4>
            <ol>
                <li>Prepare a CSV file with a column named 'Log' containing maintenance log entries</li>
                <li>Upload the file below</li>
                <li>The system will classify each entry and provide recommendations</li>
                <li>Download the results as a new CSV file</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with a 'Log' column containing maintenance log entries"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df_upload = pd.read_csv(uploaded_file)
                
                # Check if 'Log' column exists
                if 'Log' not in df_upload.columns:
                    st.markdown('<div class="error-message">❌ Error: CSV file must contain a column named "Log"</div>', unsafe_allow_html=True)
                    st.write("Available columns:", list(df_upload.columns))
                else:
                    st.markdown(f'<div class="success-message">✅ Successfully loaded {len(df_upload)} log entries</div>', unsafe_allow_html=True)
                    
                    # Show preview
                    st.markdown("### 📋 File Preview")
                    st.dataframe(df_upload.head(), use_container_width=True)
                    
                    # Process button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        process_button = st.button("🚀 Process Batch Classification", type="primary", use_container_width=True)
                    
                    if process_button:
                        with st.spinner("🔄 Processing batch classification..."):
                            # Process the batch
                            results_df = process_batch_classification(df_upload, model, vectorizer)
                            
                            # Display results
                            st.markdown("### 📊 Classification Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="maintenance_classification_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Summary statistics in cards
                            st.markdown("### 📈 Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Total Entries</h4>
                                    <h2 style="color: #667eea;">{len(results_df)}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                category_counts = results_df['Predicted Category'].value_counts()
                                most_common = category_counts.index[0] if len(category_counts) > 0 else "N/A"
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Most Common Issue</h4>
                                    <h2 style="color: #667eea;">{most_common}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                avg_confidence = results_df['Confidence (%)'].str.rstrip('%').astype(float).mean()
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Average Confidence</h4>
                                    <h2 style="color: #667eea;">{avg_confidence:.1f}%</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Category distribution chart
                            if len(category_counts) > 0:
                                st.markdown("### 📊 Category Distribution")
                                fig = create_category_distribution(category_counts, recommendations)
                                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error reading CSV file: {str(e)}</div>', unsafe_allow_html=True)
    
    elif page == "About":
        st.markdown("## ℹ️ About This Application")
        
        st.markdown("""
        <div class="metric-card">
            <h3>Aircraft Maintenance Log Classification System</h3>
            <p>This application uses advanced machine learning to automatically classify aircraft maintenance log entries into predefined categories and provide actionable recommendations for maintenance personnel.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Categories and recommendations
        st.markdown("### 🎯 Categories & Recommendations")
        
        for category, info in recommendations.items():
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {info['color']};">{category}</h4>
                <p><strong>Recommended Action:</strong> {info['action']}</p>
                <p><strong>Priority:</strong> <span style="color: {info['priority_color']}; font-weight: bold;">{info['priority']}</span></p>
                <p><strong>Details:</strong> {info['details']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Technology stack
        st.markdown("### 🛠️ Technology Stack")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Frontend & UI</h4>
                <ul>
                    <li>Streamlit</li>
                    <li>Plotly (Interactive Charts)</li>
                    <li>Custom CSS Styling</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Machine Learning</h4>
                <ul>
                    <li>Scikit-learn</li>
                    <li>NLTK (Text Processing)</li>
                    <li>TF-IDF Vectorization</li>
                    <li>Logistic Regression</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Features
        st.markdown("### ✨ Key Features")
        st.markdown("""
        <div class="metric-card">
            <ul>
                <li><strong>Real-time Classification:</strong> Instant text classification with confidence scores</li>
                <li><strong>Actionable Recommendations:</strong> Specific maintenance actions for each category</li>
                <li><strong>Batch Processing:</strong> Upload CSV files for bulk classification</li>
                <li><strong>Interactive Visualizations:</strong> Beautiful charts and analytics</li>
                <li><strong>Modern UI:</strong> Professional, responsive design</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔧 Technical Details"):
            st.markdown(
"""
### 📊 Dataset Information
- **Total Entries:** 200 synthetic maintenance log entries
- **Categories:** 4 maintenance issue types
- **Distribution:** 50 samples per category (balanced)
- **Content:** Realistic aviation terminology with aircraft identifiers
- **Features:** Aircraft ID, maintenance log text, severity levels, action items

### 🔄 Text Preprocessing Pipeline
1. **Text Normalization:** Convert to lowercase for consistency
2. **Character Cleaning:** Remove special characters and extra whitespace
3. **Tokenization:** Split text into individual words using NLTK
4. **Stop Word Removal:** Remove common English stop words
5. **Length Filtering:** Keep only tokens with length > 2 characters

### 🤖 Model Architecture
- **Primary Algorithm:** Logistic Regression (best performing)
- **Alternative Model:** Random Forest Classifier (for comparison)
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Feature Parameters:**
    - Max Features: 1,000
    - N-gram Range: (1, 2) - unigrams and bigrams
    - Min Document Frequency: 2
    - Max Document Frequency: 95%
- **Training Split:** 80% training, 20% testing (stratified)
- **Random State:** 42 (for reproducibility)

### 📈 Model Performance
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Cross-Validation:** Stratified train-test split
- **Confidence Scoring:** Probability distribution across all categories
- **Real-time Inference:** Fast prediction with confidence scores

### 🏗️ System Architecture
- **Frontend:** Streamlit with custom CSS styling
- **Visualization:** Plotly interactive charts
- **Data Processing:** Pandas for data manipulation
- **Model Persistence:** Joblib for model serialization
- **Text Processing:** NLTK for natural language processing
- **Machine Learning:** Scikit-learn for classification

### 🔧 Implementation Details
- **Data Generation:** Synthetic data with realistic aviation terminology
- **Context Phrases:** 15 different operational contexts (preflight, takeoff, landing, etc.)
- **Aircraft Types:** 8 different aircraft identifiers (A320, B737, A380, etc.)
- **Severity Levels:** Minor, Moderate, Major classifications
- **Action Items:** Maintenance crew notifications, technician inspections

### 🚀 Deployment Features
- **Single Classification:** Real-time text analysis with confidence scores
- **Batch Processing:** CSV upload and bulk classification
- **Results Export:** Download classification results as CSV
- **Interactive Analytics:** Dynamic charts and performance metrics
- **Responsive Design:** Works on desktop and mobile devices
"""
    )

if __name__ == "__main__":
    main() 