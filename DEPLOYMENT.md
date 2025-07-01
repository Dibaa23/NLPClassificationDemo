# Deployment Guide for Streamlit Cloud

This guide will help you deploy your Aircraft Maintenance Log Classification System on Streamlit Cloud.

## 🚀 Prerequisites

1. **GitHub Account**: Make sure you have a GitHub account
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Git**: Ensure Git is installed on your system

## 📋 Deployment Steps

### Step 1: Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Aircraft Maintenance Log Classifier"

# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/NLPClassificationDemo.git

# Push to GitHub
git push -u origin main
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it `NLPClassificationDemo`
3. Make it public (Streamlit Cloud requires public repos for free tier)
4. Don't initialize with README (we already have one)

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the details:
   - **Repository**: `YOUR_USERNAME/NLPClassificationDemo`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy!"

## 🔧 Configuration Files

The following files are already configured for deployment:

### `.streamlit/config.toml`
- Optimized settings for production
- Disabled development mode
- Configured for headless operation

### `packages.txt`
- System dependencies for NLTK

### `requirements.txt`
- All Python dependencies with specific versions

## 📁 Required Files for Deployment

Ensure these files are in your repository:

```
NLPClassificationDemo/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System dependencies
├── .streamlit/config.toml             # Streamlit configuration
├── maintenance_classifier_model.pkl    # Trained model
├── tfidf_vectorizer.pkl               # TF-IDF vectorizer
├── maintenance_logs_dataset.csv       # Dataset
└── README.md                          # Project documentation
```

## 🚨 Important Notes

### Model Files
- The `.pkl` files (model and vectorizer) are included in the repository
- These are required for the app to function
- They're relatively small and safe to include

### NLTK Data
- The app automatically downloads required NLTK data on first run
- This may cause a slight delay on the first deployment

### Memory and Performance
- The app is optimized for Streamlit Cloud's free tier
- Uses caching to improve performance
- Minimal memory footprint

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check that versions are compatible

2. **Model Loading Issues**
   - Verify `.pkl` files are in the repository
   - Check file paths in the code

3. **NLTK Data Issues**
   - The app handles NLTK data download automatically
   - First run may take longer

4. **Memory Issues**
   - The app uses caching to optimize memory usage
   - If issues persist, consider upgrading to paid tier

### Performance Optimization

1. **Caching**: The app uses `@st.cache_data` and `@st.cache_resource`
2. **Lazy Loading**: Models are loaded only when needed
3. **Efficient Processing**: Batch processing for large datasets

## 🌐 Post-Deployment

### Custom Domain (Optional)
- Streamlit Cloud provides a default URL
- You can configure a custom domain in the settings

### Monitoring
- Check the app logs in Streamlit Cloud dashboard
- Monitor usage and performance metrics

### Updates
- Push changes to GitHub to automatically redeploy
- Streamlit Cloud will rebuild the app automatically

## 📊 Analytics

Streamlit Cloud provides:
- Usage analytics
- Performance metrics
- Error logs
- User engagement data

## 🔒 Security Considerations

1. **Public Repository**: Free tier requires public repos
2. **No Sensitive Data**: Ensure no API keys or secrets are in the code
3. **Model Security**: The trained model is safe to share publicly

## 🎯 Next Steps

After successful deployment:

1. **Test the App**: Verify all features work correctly
2. **Share the URL**: Share with stakeholders and users
3. **Monitor Usage**: Track performance and user feedback
4. **Iterate**: Make improvements based on user feedback

## 📞 Support

If you encounter issues:

1. Check Streamlit Cloud documentation
2. Review app logs in the dashboard
3. Test locally first: `streamlit run app.py`
4. Check GitHub issues for similar problems

---

**Happy Deploying! 🚀** 