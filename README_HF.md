---
title: reLAISS - Supernova Similarity Search
emoji: 🌟
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# reLAISS: Supernova Similarity Search

Find similar supernovae using light curve and host galaxy features! This interactive tool uses the reLAISS (re-implementation of LAISS) algorithm to search through 25,000+ ZTF transients.

## Features

- **Fast similarity search** using approximate nearest neighbors (NGT)
- **Flexible feature selection** - Choose from 25 light curve features and 18 host galaxy properties
- **Interactive visualizations** - Compare light curves and analyze features
- **Real-time data** - Fetches metadata from ANTARES and TNS APIs

## Usage

1. **Enter a ZTF Object ID** (e.g., `ZTF21abbzjeq`)
2. **Select features** using the Quick Presets or customize your selection
3. **Click Search** to find similar transients
4. **Explore results** across multiple tabs:
   - Summary: Overview cards for each match
   - Light Curves: Visual comparison of photometry
   - Feature Analysis: Detailed feature breakdown
   - Host Galaxies: Host galaxy images and properties

## Performance

- **First search**: ~90 seconds (one-time preprocessing + caching)
- **Subsequent searches**: ~3-5 seconds
- **Index rebuilds** (when changing features): ~10-15 seconds

## Citation

If you use reLAISS in your research, please cite:

```
Reynolds et al. (2024) - reLAISS: A re-implementation of LAISS for transient similarity search
```

## Links

- [GitHub Repository](https://github.com/evan-reynolds/re-laiss/)
- [Original LAISS Paper](https://arxiv.org/abs/2109.01665)
