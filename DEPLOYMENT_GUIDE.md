# Hugging Face Spaces Deployment Guide

## Prerequisites

1. Create a Hugging Face account at https://huggingface.co
2. Ensure you have all required files ready

## Required Files

Your Hugging Face Space needs these files:

```
re-laiss/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README_HF.md                    # Space description (rename to README.md)
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── src/
│   └── relaiss/                   # reLAISS package
│       ├── __init__.py
│       ├── relaiss.py
│       ├── features.py
│       ├── fetch.py
│       ├── search.py
│       └── utils.py
├── data/
│   └── LAISS_dataset_bank.csv    # Reference dataset
└── sfddata-master/                # Dust map data
```

## Deployment Steps

### Option 1: Git Repository (Recommended)

1. **Initialize Git repository** (if not already done):
   ```bash
   git init
   git add app.py requirements.txt README_HF.md .streamlit/ src/ data/ sfddata-master/
   git commit -m "Initial commit for HF Space"
   ```

2. **Create a new Space** on Hugging Face:
   - Go to https://huggingface.co/new-space
   - Choose a name (e.g., `relaiss-search`)
   - Select **Streamlit** as the SDK
   - Choose **Public** or **Private**
   - Click "Create Space"

3. **Push to Hugging Face**:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf main
   ```

4. **Rename README_HF.md**:
   - In the HF web interface, rename `README_HF.md` to `README.md`
   - This ensures the YAML frontmatter is recognized

### Option 2: Web Upload

1. **Create a new Space** on Hugging Face (see above)

2. **Upload files** via the web interface:
   - Click "Files" → "Add file" → "Upload files"
   - Upload all required files and folders
   - **Important**: Rename `README_HF.md` to `README.md`

3. **Wait for build**:
   - HF will automatically install dependencies
   - First build takes ~5-10 minutes

## Storage Considerations

- **Dataset**: `LAISS_dataset_bank.csv` (~50MB)
- **Dust maps**: `sfddata-master/` (~500MB)
- **Cache**: After first run, NGT index cache (~100MB)
- **Total**: ~650MB (well within free tier limits)

## Environment Variables

No environment variables are required! The app uses:
- Cached preprocessing (saved to persistent storage)
- Public APIs (ANTARES, TNS) - no auth needed

## Performance Optimization

The app is optimized for HF Spaces:

1. **Caching**: Uses `~/.relaiss/cache` for persistent storage
2. **Lazy loading**: Only loads data when first search is performed
3. **Parallel fetching**: API calls run concurrently
4. **Efficient indexing**: NGT provides fast approximate nearest neighbors

### Expected Performance

- **Cold start** (first visitor after idle): ~90 seconds
- **Warm start** (with cached index): ~3-5 seconds per search
- **Index rebuild** (changing features): ~10-15 seconds

## Troubleshooting

### Build fails with "Could not find sfdmap2"

- Check Python version in Space settings (should be ≥3.9)
- If using Python 3.8, change `sfdmap2` to `sfdmap` in requirements.txt

### App shows "ModuleNotFoundError: No module named 'relaiss'"

- Ensure `src/relaiss/` directory structure is correct
- Check that `__init__.py` exists in `src/relaiss/`

### Slow first search

- This is expected! The first search triggers:
  - Loading 25,515 transients from CSV
  - Feature engineering with KNN imputation (~90s)
  - Building NGT index
- Subsequent searches are fast (~3-5s)

### Out of memory errors

- Try reducing the number of features in the default selection
- HF Spaces free tier has 16GB RAM, which should be sufficient

## Monitoring

Check your Space's logs:
1. Go to your Space on HF
2. Click "Logs" tab
3. Look for `[TIMER]` messages showing performance metrics

## Updating the Space

To update your deployed app:

```bash
git add <changed-files>
git commit -m "Update description"
git push hf main
```

HF will automatically rebuild and redeploy.

## Support

- HF Spaces docs: https://huggingface.co/docs/hub/spaces
- Streamlit docs: https://docs.streamlit.io
- reLAISS issues: https://github.com/evan-reynolds/re-laiss/issues
