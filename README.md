# Aircraft Maintenance Log Classification System

A complete end-to-end NLP text classification application for categorizing aircraft maintenance log entries using machine learning and Streamlit with a modern, polished UI.

## 🚀 Project Overview

This project demonstrates a complete machine learning pipeline for text classification with a beautiful, professional interface:

- **Data Generation**: 200 synthetic maintenance log entries across 4 categories
- **Text Preprocessing**: Advanced NLP techniques with NLTK
- **Model Training**: Logistic Regression with TF-IDF vectorization
- **Interactive Web App**: Real-time classification with modern Streamlit UI
- **Performance Analysis**: Interactive visualizations with Plotly
- **Batch Processing**: CSV upload and bulk classification capabilities

## 📋 Categories

The system classifies maintenance logs into four main categories:

1. **Hydraulic Issue** - Problems with hydraulic systems, pressure, leaks
2. **Electrical Issue** - Electrical faults, wiring problems, avionics
3. **Landing Gear Issue** - Landing gear malfunctions, tire problems
4. **Engine Issue** - Engine performance, fuel system problems

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Machine Learning**: Scikit-learn
- **Text Processing**: NLTK, TF-IDF Vectorization
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly (Interactive charts)
- **Model Persistence**: Joblib

## 🎨 UI Features

### Modern Design
- **Gradient Headers**: Beautiful gradient backgrounds with professional styling
- **Card-based Layout**: Clean, organized information display
- **Interactive Charts**: Plotly-powered visualizations with hover effects
- **Responsive Design**: Optimized for different screen sizes
- **Color-coded Categories**: Visual distinction for different maintenance issues

### Enhanced User Experience
- **Real-time Classification**: Instant results with confidence scores
- **Batch Processing**: Upload CSV files for bulk classification
- **Download Results**: Export classification results as CSV
- **Professional Styling**: Modern buttons, forms, and navigation
- **Success/Error Messages**: Clear feedback with styled notifications

## 📁 Project Structure

```
NLPClassificationDemo/
├── data_preparation_and_training.ipynb  # Jupyter notebook for training
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── maintenance_classifier_model.pkl    # Trained model (generated)
├── tfidf_vectorizer.pkl               # TF-IDF vectorizer (generated)
└── maintenance_logs_dataset.csv       # Dataset (generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the Jupyter notebook to generate the dataset and train the model:

```bash
jupyter notebook data_preparation_and_training.ipynb
```

**Or run all cells in the notebook to:**
- Generate 200 synthetic maintenance log entries
- Preprocess and vectorize the text data
- Train a Logistic Regression classifier
- Save the model and vectorizer

### 3. Launch the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Features

### Single Text Classification
- **Real-time Analysis**: Instant classification with confidence scores
- **Interactive Charts**: Plotly-powered confidence visualization
- **Actionable Recommendations**: Specific maintenance actions for each category
- **Priority Indicators**: Color-coded priority levels (High/Critical)
- **Text Preprocessing**: Expandable details showing cleaned text

### Batch CSV Processing
- **File Upload**: Drag-and-drop CSV file upload
- **Bulk Classification**: Process multiple entries simultaneously
- **Results Export**: Download classification results as CSV
- **Summary Statistics**: Key metrics and category distribution
- **Interactive Analytics**: Pie charts and performance metrics

### Modern UI Components
- **Gradient Headers**: Professional styling with aviation theme
- **Metric Cards**: Clean display of statistics and results
- **Category Cards**: Color-coded classification results
- **Recommendation Boxes**: Styled action items with priority indicators
- **Success/Error Messages**: Clear feedback with gradient styling

## 🔧 Technical Implementation

### Data Generation
- 200 synthetic maintenance log entries
- Realistic aviation terminology
- Balanced distribution across categories
- Contextual phrases and aircraft identifiers

### Text Preprocessing
1. Text normalization (lowercase)
2. Special character removal
3. Tokenization using NLTK
4. Stop word removal
5. TF-IDF vectorization (max 1000 features, 1-2 grams)

### Model Training
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectors
- **Validation**: Stratified train-test split (80/20)
- **Performance**: High accuracy across all categories

### Web Application
- **Framework**: Streamlit with custom CSS
- **Layout**: Wide layout with sidebar navigation
- **Caching**: Optimized model loading and data processing
- **Visualization**: Interactive Plotly charts
- **Styling**: Modern gradient designs and hover effects

## 📈 Model Performance

The trained model achieves:
- **High accuracy** across all maintenance categories
- **Balanced performance** for each class
- **Fast inference** for real-time classification
- **Robust preprocessing** for various input formats

## 🎯 Usage Examples

### Example 1: Hydraulic Issue
```
Input: "Aircraft A320-001: hydraulic pressure low during preflight inspection. Severity: Moderate."
Output: Hydraulic Issue (High confidence)
Recommendation: Inspect hydraulic pumps and fluid levels.
```

### Example 2: Electrical Issue
```
Input: "Aircraft B737-002: electrical system failure during takeoff. Avionics malfunction detected."
Output: Electrical Issue (High confidence)
Recommendation: Check auxiliary power systems and circuit breakers.
```

### Example 3: Landing Gear Issue
```
Input: "Aircraft A380-003: landing gear not retracting after takeoff. Warning light illuminated."
Output: Landing Gear Issue (High confidence)
Recommendation: Examine landing gear retraction and extension systems.
```

## 🔍 Advanced Features

### Interactive Visualizations
- **Confidence Charts**: Bar charts showing prediction confidence across categories
- **Category Distribution**: Interactive pie charts for batch processing results
- **Hover Effects**: Detailed information on chart interaction
- **Responsive Design**: Charts adapt to different screen sizes

### Batch Processing
- **CSV Upload**: Support for bulk classification
- **Progress Tracking**: Real-time processing feedback
- **Results Summary**: Key statistics and metrics
- **Export Functionality**: Download results in CSV format

### Professional Styling
- **Custom CSS**: Modern gradient designs and animations
- **Color Themes**: Aviation-inspired color palette
- **Typography**: Professional font styling
- **Layout**: Clean, organized information hierarchy

## 🛠️ Customization

### Adding New Categories
1. Modify the `categories` dictionary in the notebook
2. Add new keywords and phrases
3. Retrain the model
4. Update the app's category colors and styling

### Model Improvements
- Experiment with different algorithms (Random Forest, SVM)
- Adjust TF-IDF parameters
- Add more training data
- Implement cross-validation

### UI Enhancements
- Customize color schemes
- Add more interactive elements
- Implement dark mode
- Add animation effects

## 📝 Requirements

- Python 3.7+
- Jupyter Notebook
- Internet connection (for NLTK data download)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the application
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Streamlit team for the excellent web app framework
- Scikit-learn for machine learning capabilities
- NLTK for natural language processing tools
- Plotly for interactive visualizations

## 🆘 Troubleshooting

### Common Issues

**Model files not found:**
- Ensure you've run the Jupyter notebook completely
- Check that `maintenance_classifier_model.pkl` and `tfidf_vectorizer.pkl` exist

**NLTK data missing:**
- The app will automatically download required NLTK data
- If issues persist, manually download: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

**Streamlit not starting:**
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the Jupyter notebook for detailed explanations
3. Examine the code comments for implementation details

---