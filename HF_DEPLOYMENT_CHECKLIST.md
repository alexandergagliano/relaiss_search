# Hugging Face Spaces Deployment Checklist

## ✅ Pre-Deployment Checklist

### Files Created/Updated
- [x] `requirements.txt` - Updated with all dependencies
- [x] `README_HF.md` - Created with HF Space YAML frontmatter
- [x] `.streamlit/config.toml` - Created with theme and server settings
- [x] `.gitignore` - Created to exclude unnecessary files
- [x] `DEPLOYMENT_GUIDE.md` - Step-by-step deployment instructions

### Files Required for Deployment
- [ ] `app.py` - Main Streamlit application
- [ ] `src/relaiss/` - Package directory with all modules
- [ ] `data/LAISS_dataset_bank.csv` - Reference dataset
- [ ] `sfddata-master/` - Dust map data folder
- [ ] `requirements.txt` - Python dependencies
- [ ] `README_HF.md` - Space README (will be renamed to README.md)

## 📦 Deployment Steps

### Quick Start (Git Method)
```bash
# 1. Initialize git (if not already done)
git init

# 2. Add files
git add app.py requirements.txt README_HF.md .streamlit/ src/ data/ sfddata-master/

# 3. Commit
git commit -m "Deploy reLAISS to HF Spaces"

# 4. Create Space on HF website
# Visit: https://huggingface.co/new-space
# - Name: relaiss-search (or your choice)
# - SDK: Streamlit
# - Visibility: Public

# 5. Add HF remote and push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main

# 6. Rename README_HF.md to README.md in HF web interface
```

## 🔍 Post-Deployment Verification

### After deployment completes:
- [ ] Space builds successfully (check logs)
- [ ] App loads without errors
- [ ] First search completes (~90 seconds expected)
- [ ] Subsequent searches are fast (~3-5 seconds)
- [ ] All tabs (Summary, Light Curves, Feature Analysis, Host Galaxies) work
- [ ] Query object card displays correctly
- [ ] Match cards display with proper styling
- [ ] Preset buttons (Light Curve, Host, All) work correctly

## 📊 Performance Benchmarks

Expected timings on HF Spaces:
- **First load** (cold start): ~90 seconds
- **Cached search**: 3-5 seconds
- **Index rebuild**: 10-15 seconds

## 🐛 Common Issues

### Build Failures
- Python version mismatch → Check Space settings (use Python 3.9+)
- Missing dependencies → Verify requirements.txt is complete

### Runtime Issues  
- ModuleNotFoundError → Check src/relaiss structure
- Slow performance → Expected on first run (90s for preprocessing)
- Memory errors → Unlikely with free tier (16GB RAM)

## 📝 Notes

- The `.relaiss` cache directory will be created automatically on HF
- Persistent storage ensures the index survives Space restarts
- No API keys or environment variables needed!
- Free tier is sufficient for this app

## 🎉 Success!

Once deployed, your Space will be available at:
`https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

Share it with the world! 🌟
