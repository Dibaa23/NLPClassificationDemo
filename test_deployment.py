#!/usr/bin/env python3
"""
Test script to verify deployment readiness
Run this before deploying to Streamlit Cloud
"""

import os
import sys
import joblib
import pandas as pd
import nltk

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_files():
    """Test if all required files exist"""
    print("\n📁 Testing required files...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'maintenance_classifier_model.pkl',
        'tfidf_vectorizer.pkl',
        'maintenance_logs_dataset.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def test_model_loading():
    """Test if model files can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        model = joblib.load('maintenance_classifier_model.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("✅ Vectorizer loaded successfully")
    except Exception as e:
        print(f"❌ Vectorizer loading failed: {e}")
        return False
    
    return True

def test_dataset_loading():
    """Test if dataset can be loaded"""
    print("\n📊 Testing dataset loading...")
    
    try:
        df = pd.read_csv('maintenance_logs_dataset.csv')
        print(f"✅ Dataset loaded successfully ({len(df)} rows)")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    return True

def test_nltk_data():
    """Test if NLTK data is available"""
    print("\n📚 Testing NLTK data...")
    
    try:
        # Test punkt tokenizer
        nltk.data.find('tokenizers/punkt')
        print("✅ NLTK punkt tokenizer available")
    except LookupError:
        print("⚠️  NLTK punkt tokenizer not found (will be downloaded)")
    
    try:
        # Test stopwords
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK stopwords available")
    except LookupError:
        print("⚠️  NLTK stopwords not found (will be downloaded)")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Aircraft Maintenance Log Classifier - Deployment Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_files,
        test_model_loading,
        test_dataset_loading,
        test_nltk_data
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    if all(results):
        print("🎉 All tests passed! Your app is ready for deployment.")
        print("\nNext steps:")
        print("1. Push your code to GitHub")
        print("2. Go to share.streamlit.io")
        print("3. Deploy your app!")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 