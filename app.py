import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import io
from PIL import Image
import requests
from pathlib import Path

# Add the reLAISS package to the path
sys.path.append(str(Path(__file__).parent.parent / "re-laiss" / "src"))

import relaiss as rl
from relaiss.constants import lc_features_const, host_features_const, anom_lc_features_const
from relaiss.search import primer
from relaiss.fetch import fetch_ps1_rgb_jpeg, fetch_ps1_cutout
import antares_client
from astropy.visualization import AsinhStretch, PercentileInterval

# Page configuration
st.set_page_config(
    page_title="reLAISS - Astronomical Transient Similarity Search",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for elegant dark styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        color: #e1e5e9;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    .metric-card {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid #4299e1;
        font-family: 'Inter', sans-serif;
    }
    .stButton > button {
        width: 100%;
        background-color: #4299e1;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #3182ce;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
    }
    .stSelectbox > div > div {
        font-family: 'Inter', sans-serif;
    }
    .stTextInput > div > div > input {
        font-family: 'Inter', sans-serif;
    }
    .stSlider > div > div > div > div {
        font-family: 'Inter', sans-serif;
    }
    .stCheckbox > div > label {
        font-family: 'Inter', sans-serif;
    }
    .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    .stDataFrame {
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'relaiss_client' not in st.session_state:
    st.session_state.relaiss_client = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'primer_dict' not in st.session_state:
    st.session_state.primer_dict = None
if 'preset_basic' not in st.session_state:
    st.session_state.preset_basic = False
if 'preset_full' not in st.session_state:
    st.session_state.preset_full = False
if 'selected_lc_features' not in st.session_state:
    st.session_state.selected_lc_features = ['g_peak_time', 'r_peak_time']
if 'selected_host_features' not in st.session_state:
    st.session_state.selected_host_features = []  # Start with no host features for faster initial load
if 'indexed_lc_features' not in st.session_state:
    st.session_state.indexed_lc_features = None
if 'indexed_host_features' not in st.session_state:
    st.session_state.indexed_host_features = None

def initialize_relaiss():
    """Initialize the reLAISS client."""
    try:
        # Get current feature selection
        lc_features = st.session_state.selected_lc_features
        host_features = st.session_state.selected_host_features

        # Check if we need to rebuild the index
        need_rebuild = False
        if (st.session_state.indexed_lc_features != lc_features or
            st.session_state.indexed_host_features != host_features):
            need_rebuild = True

        status_msg = "Rebuilding index for new features..." if need_rebuild else "Loading reference bank and building index..."

        with st.spinner(status_msg):
            # Clear only index cache if we need to rebuild (preserve preprocessed dataset bank)
            if need_rebuild:
                from relaiss.utils import get_cache_dir
                import shutil
                from pathlib import Path

                cache_dir = Path(get_cache_dir())
                index_dir = cache_dir / 'indices'

                if index_dir.exists():
                    try:
                        shutil.rmtree(index_dir)
                    except Exception as e:
                        st.error(f"Could not clear index cache: {e}")
                        return False

            client = rl.ReLAISS()

            try:
                client.load_reference(
                    path_to_sfd_folder='./sfddata-master',
                    lc_features=lc_features,
                    host_features=host_features,
                    use_pca=False,
                    force_recreation_of_index=need_rebuild
                )

                # Track the features used for this index
                st.session_state.indexed_lc_features = lc_features.copy()
                st.session_state.indexed_host_features = host_features.copy()
                st.session_state.relaiss_client = client
                return True

            except Exception as e:
                st.error(f"Error: {str(e)}")
                return False

    except Exception as e:
        st.error(f"Error initializing reLAISS: {str(e)}")
        return False

def search_similar_transients(ztf_id, n_matches, lc_features, host_features):
    """Search for similar transients using reLAISS."""
    import time
    try:
        # Use the existing client (should be initialized by now)
        client = st.session_state.relaiss_client

        # Perform search
        with st.spinner(f"Searching for similar transients to {ztf_id}..."):
            search_start = time.perf_counter()
            print(f"\n[TIMER] app.search_similar_transients: Starting search for {ztf_id}", flush=True)

            results = client.find_neighbors(
                ztf_object_id=ztf_id,
                n=n_matches,
                plot=False,
                save_figures=False
            )

            search_end = time.perf_counter()
            search_elapsed = search_end - search_start
            print(f"[TIMER] app.search_similar_transients: TOTAL TIME = {search_elapsed:.3f} seconds", flush=True)

            return results

    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return None



def create_lightcurve_plot(ztf_id, timeseries_data, title=None):
    """Create an interactive light curve plot."""
    if timeseries_data is None or timeseries_data.empty:
        return None
    
    # Filter data by band
    g_data = timeseries_data[timeseries_data['ant_passband'] == 'g']
    r_data = timeseries_data[timeseries_data['ant_passband'] == 'R']
    
    fig = go.Figure()
    
    # Add g-band data
    if not g_data.empty:
        fig.add_trace(go.Scatter(
            x=g_data['ant_mjd'],
            y=g_data['ant_mag'],
            mode='markers+lines',
            name='g-band',
            marker=dict(color='green', size=6),
            error_y=dict(type='data', array=g_data['ant_magerr'], visible=True)
        ))
    
    # Add r-band data
    if not r_data.empty:
        fig.add_trace(go.Scatter(
            x=r_data['ant_mjd'],
            y=r_data['ant_mag'],
            mode='markers+lines',
            name='r-band',
            marker=dict(color='red', size=6),
            error_y=dict(type='data', array=r_data['ant_magerr'], visible=True)
        ))
    
    # Use custom title if provided, otherwise default
    plot_title = title if title else f"Light Curve for {ztf_id}"
    
    fig.update_layout(
        title=plot_title,
        xaxis_title="MJD",
        yaxis_title="Magnitude",
        yaxis=dict(autorange="reversed"),
        height=400,
        showlegend=True
    )
    
    return fig

def create_feature_comparison_plot(query_features, neighbor_features, feature_names):
    """Create a feature comparison plot."""
    fig = go.Figure()
    
    # Add query object
    fig.add_trace(go.Bar(
        x=feature_names,
        y=query_features,
        name='Query Object',
        marker_color='#1f77b4'
    ))
    
    # Add neighbors
    for i, neighbor in enumerate(neighbor_features):
        fig.add_trace(go.Bar(
            x=feature_names,
            y=neighbor,
            name=f'Neighbor {i+1}',
            marker_color=f'rgba(255, 127, 14, {0.7 - i*0.1})'
        ))
    
    fig.update_layout(
        title="Feature Comparison",
        xaxis_title="Features",
        yaxis_title="Normalized Values",
        barmode='group',
        height=400
    )
    
    return fig

def fetch_host_image(ra, dec, size_pix=100):
    """Fetch a host galaxy image from PanSTARRS."""
    try:
        # Try RGB first, fallback to grayscale
        try:
            img_array = fetch_ps1_rgb_jpeg(ra, dec, size_pix=size_pix)
            img = Image.fromarray(img_array)
        except:
            img_array = fetch_ps1_cutout(ra, dec, size_pix=size_pix, flt="r")
            img = Image.fromarray(img_array)
        
        # Convert to bytes for Streamlit
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    except Exception as e:
        st.warning(f"Could not fetch image for coordinates ({ra:.4f}, {dec:.4f}): {str(e)}")
        return None

def main():
    # Header with logo
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <img src="https://github.com/evan-reynolds/re-laiss/raw/main/static/reLAISS_logo.png" 
             alt="reLAISS Logo" 
             style="max-width: 300px; height: auto; margin-bottom: 1rem;">
        <p style="text-align: center; font-size: 1.2rem; color: #a0aec0; font-family: 'Inter', sans-serif;">Reference Lightcurve Anomaly and Similarity Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## Search Parameters")


    
    # Cache management
    with st.sidebar.expander("Cache Management", expanded=False):
        st.markdown("**Current Status:**")
        if st.session_state.relaiss_client is not None:
            st.success("Index built and ready")
        else:
            st.warning("No index built")

        if st.button("Clear Cache", help="Clear all cached data to fix index corruption issues"):
            from relaiss.utils import get_cache_dir
            import shutil

            cache_dir = get_cache_dir()
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    st.success("Cache cleared successfully!")
                    st.session_state.relaiss_client = None
                    st.session_state.indexed_lc_features = None
                    st.session_state.indexed_host_features = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
            else:
                st.info("No cache to clear.")
    
    # Search parameters
    st.sidebar.markdown("### Search Configuration")
    
    # ZTF ID input
    ztf_id = st.sidebar.text_input(
        "ZTF Object ID",
        value="ZTF21abbzjeq",
        help="Enter a valid ZTF transient ID"
    )
    
    # Number of matches
    n_matches = st.sidebar.slider(
        "Number of Matches",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of similar transients to find"
    )
    
    # Feature selection
    st.sidebar.markdown("### Feature Selection")
    
    # Available features from reLAISS constants
    # Exclude anomaly features (multiple peak features) as they cause imputation issues
    available_lc_features = lc_features_const  # Only use stable lightcurve features
    available_host_features = host_features_const
    
    # Lightcurve features
    # Filter session state to only include available features
    if 'selected_lc_features' in st.session_state:
        valid_lc_features = [f for f in st.session_state.selected_lc_features if f in available_lc_features]
        if valid_lc_features != st.session_state.selected_lc_features:
            st.session_state.selected_lc_features = valid_lc_features
    else:
        st.session_state.selected_lc_features = ['g_peak_time', 'r_peak_time']
    
    lc_count = len(st.session_state.selected_lc_features)
    with st.sidebar.expander(f"Light Curve Features ({lc_count})", expanded=False):
        selected_lc_features = st.multiselect(
            "Select light curve features",
            options=available_lc_features,
            default=st.session_state.selected_lc_features,
            help="Choose which light curve features to include in the similarity search",
            key="lc_features_select"
        )
        st.session_state.selected_lc_features = selected_lc_features
    
    # Host galaxy features
    # Filter session state to only include available features
    if 'selected_host_features' in st.session_state:
        valid_host_features = [f for f in st.session_state.selected_host_features if f in available_host_features]
        if valid_host_features != st.session_state.selected_host_features:
            st.session_state.selected_host_features = valid_host_features
    else:
        st.session_state.selected_host_features = []
    
    host_count = len(st.session_state.selected_host_features)
    with st.sidebar.expander(f"Host Galaxy Features ({host_count})", expanded=False):
        selected_host_features = st.multiselect(
            "Select host galaxy features",
            options=available_host_features,
            default=st.session_state.selected_host_features,
            help="Choose which host galaxy features to include in the similarity search",
            key="host_features_select"
        )
        st.session_state.selected_host_features = selected_host_features
    
    # Quick preset feature sets
    st.sidebar.markdown("**Quick Presets:**")
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        if st.button("Light Curve", help="All stable light curve features", use_container_width=True):
            st.session_state.selected_lc_features = available_lc_features
            st.session_state.selected_host_features = []
            st.rerun()

    with col2:
        if st.button("Host", help="Host galaxy features only", use_container_width=True):
            st.session_state.selected_lc_features = []
            st.session_state.selected_host_features = available_host_features
            st.rerun()

    with col3:
        if st.button("All", help="All features (light curve + host)", use_container_width=True):
            st.session_state.selected_lc_features = available_lc_features
            st.session_state.selected_host_features = available_host_features
            st.rerun()
    
    # Search button
    if st.sidebar.button("Search Similar Transients", type="primary"):
        if not ztf_id:
            st.error("Please enter a ZTF Object ID!")
        elif not selected_lc_features and not selected_host_features:
            st.error("Please select at least one feature!")
        else:
            # Clear old results immediately
            st.session_state.search_results = None

            # Check if we need to initialize or rebuild
            need_initialize = st.session_state.relaiss_client is None
            need_rebuild = False
            
            if not need_initialize and st.session_state.relaiss_client is not None:
                # Check if features changed
                current_lc = st.session_state.indexed_lc_features
                current_host = st.session_state.indexed_host_features
                if (current_lc != selected_lc_features or current_host != selected_host_features):
                    need_rebuild = True

            # Initialize or rebuild if needed
            if need_initialize or need_rebuild:
                if need_rebuild:
                    st.info("Feature selection changed - rebuilding index...")
                    st.session_state.relaiss_client = None
                    st.session_state.indexed_lc_features = None
                    st.session_state.indexed_host_features = None

                if not initialize_relaiss():
                    st.error("Failed to initialize reLAISS!")
                    return

            # Perform search
            results = search_similar_transients(
                ztf_id, n_matches, selected_lc_features, selected_host_features
            )

            if results is not None:
                st.session_state.search_results = results
    
    # Main content area
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        
        # Results header
        st.markdown('<h2 class="sub-header">Search Results</h2>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Light Curves", "Feature Analysis", "Host Galaxies"])
        
        with tab1:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", len(results))
            with col2:
                st.metric("Best Match Distance", f"{results['dist'].iloc[0]:.4f}")
            with col3:
                st.metric("Features Used", len(selected_lc_features) + len(selected_host_features))

            # Query object card
            st.markdown("### Query Object")
            query_ztf_id = results['input_ztf_id'].iloc[0]

            # Fetch query object metadata
            try:
                from antares_client._api.models import Locus
                from relaiss.fetch import get_TNS_data
                import antares_client

                query_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=query_ztf_id)
                query_iau, query_spec, query_z = get_TNS_data(query_ztf_id)
            except Exception as e:
                query_iau, query_spec, query_z = 'N/A', 'N/A', 'N/A'

            # Display query object card with green accent
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2f4f3f 0%, #1a2f2a 100%);
                        padding: 0.625rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                        border-left: 3px solid #48bb78;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.375rem;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #fff;">Query: {query_ztf_id}</div>
                    <div style="background-color: #48bb78; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; color: #fff; font-weight: 600;">
                        INPUT
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; color: #e2e8f0;">
                    <div>
                        <div style="font-size: 0.65rem; color: #a0aec0; margin-bottom: 0.15rem;">IAU Name</div>
                        <div style="font-weight: 500; font-size: 0.85rem;">{query_iau}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.65rem; color: #a0aec0; margin-bottom: 0.15rem;">Classification</div>
                        <div style="font-weight: 500; font-size: 0.85rem;">{query_spec}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.65rem; color: #a0aec0; margin-bottom: 0.15rem;">Redshift</div>
                        <div style="font-weight: 500; font-size: 0.85rem;">{query_z if isinstance(query_z, str) else f'{query_z:.4f}' if query_z != 'N/A' else 'N/A'}</div>
                    </div>
                </div>
                <div style="margin-top: 0.375rem;">
                    <a href="https://alerce.online/object/{query_ztf_id}" target="_blank"
                       style="color: #68d391; text-decoration: none; font-size: 0.75rem; font-weight: 500;">
                        View on ALeRCE →
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Matched Transients")

            # Display each match as a card
            for i, (ztf_id_match, distance) in enumerate(zip(results['neighbor_ztf_id'], results['dist'])):
                iau_name = results['iau_name'].iloc[i] if 'iau_name' in results.columns else 'N/A'
                spec = results['spec_cls'].iloc[i] if 'spec_cls' in results.columns else 'N/A'
                z = results['z'].iloc[i] if 'z' in results.columns else 'N/A'

                # Create a styled card for each match
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
                            padding: 0.625rem;
                            border-radius: 8px;
                            margin-bottom: 0.5rem;
                            border-left: 3px solid {'#4299e1' if i == 0 else '#718096'};
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.375rem;">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #fff;">#{i + 1} {ztf_id_match}</div>
                        <div style="background-color: #4a5568; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; color: #e2e8f0;">
                            {distance:.4f}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; color: #e2e8f0;">
                        <div>
                            <div style="font-size: 0.65rem; color: #a0aec0; margin-bottom: 0.15rem;">IAU Name</div>
                            <div style="font-weight: 500; font-size: 0.85rem;">{iau_name}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.65rem; color: #a0aec0; margin-bottom: 0.15rem;">Classification</div>
                            <div style="font-weight: 500; font-size: 0.85rem;">{spec}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.65rem; color: #a0aec0; margin-bottom: 0.15rem;">Redshift</div>
                            <div style="font-weight: 500; font-size: 0.85rem;">{z if isinstance(z, str) else f'{z:.4f}' if z != 'N/A' else 'N/A'}</div>
                        </div>
                    </div>
                    <div style="margin-top: 0.375rem;">
                        <a href="https://alerce.online/object/{ztf_id_match}" target="_blank"
                           style="color: #63b3ed; text-decoration: none; font-size: 0.75rem; font-weight: 500;">
                            View on ALeRCE →
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            # Light curve plots
            st.markdown("### Light Curve Comparison")

            # Get primer data for the query object
            if st.session_state.relaiss_client:
                try:
                    import antares_client

                    primer_dict = primer(
                        lc_ztf_id=ztf_id,
                        dataset_bank_path=st.session_state.relaiss_client.bank_csv,
                        path_to_timeseries_folder='./timeseries',
                        path_to_sfd_folder='./sfddata-master',
                        lc_features=selected_lc_features,
                        host_features=selected_host_features
                    )

                    # Query object light curve
                    st.markdown("### Query Object")
                    try:
                        query_info = antares_client.search.get_by_ztf_object_id(ztf_id)
                        query_timeseries = query_info.timeseries.to_pandas()
                        query_title = f"Query: {ztf_id} (Distance: 0.0000)"
                        query_fig = create_lightcurve_plot(ztf_id, query_timeseries, query_title)
                        if query_fig:
                            st.plotly_chart(query_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not load light curve for query object {ztf_id}: {str(e)}")

                    # Neighbor light curves
                    st.markdown("### Similar Neighbors")
                    for i, neighbor_id in enumerate(results['neighbor_ztf_id'][:3]):  # Show top 3
                        try:
                            neighbor_info = antares_client.search.get_by_ztf_object_id(neighbor_id)
                            neighbor_timeseries = neighbor_info.timeseries.to_pandas()
                            distance = results['dist'].iloc[i]
                            neighbor_title = f"Neighbor #{i+1}: {neighbor_id} (Distance: {distance:.4f})"
                            neighbor_fig = create_lightcurve_plot(neighbor_id, neighbor_timeseries, neighbor_title)
                            if neighbor_fig:
                                st.plotly_chart(neighbor_fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not load light curve for {neighbor_id}: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error loading light curve data: {str(e)}")
        
        with tab3:
            # Feature analysis
            st.markdown("### Feature Analysis")
            
            if st.session_state.relaiss_client:
                try:
                    with st.spinner("Loading feature analysis data..."):
                        # Get feature vectors for comparison
                        query_features = primer(
                            lc_ztf_id=ztf_id,
                            dataset_bank_path=st.session_state.relaiss_client.bank_csv,
                            path_to_timeseries_folder='./timeseries',
                            path_to_sfd_folder='./sfddata-master',
                            lc_features=selected_lc_features,
                            host_features=selected_host_features
                        )['locus_feat_arr']
                        
                        # Create feature comparison plot
                        all_features = selected_lc_features + selected_host_features
                        if len(all_features) > 0:
                            neighbor_features = []
                            for neighbor_id in results['neighbor_ztf_id'][:3]:
                                try:
                                    neighbor_primer = primer(
                                        lc_ztf_id=neighbor_id,
                                        dataset_bank_path=st.session_state.relaiss_client.bank_csv,
                                        path_to_timeseries_folder='./timeseries',
                                        path_to_sfd_folder='./sfddata-master',
                                        lc_features=selected_lc_features,
                                        host_features=selected_host_features
                                    )
                                    neighbor_features.append(neighbor_primer['locus_feat_arr'])
                                except:
                                    continue
                            
                            if neighbor_features:
                                comparison_fig = create_feature_comparison_plot(
                                    query_features, neighbor_features, all_features
                                )
                                st.plotly_chart(comparison_fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in feature analysis: {str(e)}")
        
        with tab4:
            st.markdown("### Host Galaxy Images")
            if st.session_state.relaiss_client is None:
                st.warning("Please initialize reLAISS first!")
            else:
                try:
                    with st.spinner("Loading host galaxy images..."):
                        ref_data = st.session_state.relaiss_client.get_preprocessed_dataframe()
                        neighbor_ids = list(results['neighbor_ztf_id'])
                        
                        # Set index exactly like internal plotting code
                        ref_data = ref_data.copy()
                        ref_data.set_index('ztf_object_id', drop=False, inplace=True)
                        
                        # Use exact same logic as internal plotting code
                        hosts_to_plot = neighbor_ids.copy()
                        host_ra_l, host_dec_l = [], []
                        
                        for ztfid in hosts_to_plot:
                            try:
                                # Try both possible column name variations (exact same as internal code)
                                if 'host_ra' in ref_data.columns and 'host_dec' in ref_data.columns:
                                    host_ra, host_dec = (
                                        ref_data.loc[ztfid].host_ra,
                                        ref_data.loc[ztfid].host_dec,
                                    )
                                elif 'raMean' in ref_data.columns and 'decMean' in ref_data.columns:
                                    host_ra, host_dec = (
                                        ref_data.loc[ztfid].raMean,
                                        ref_data.loc[ztfid].decMean,
                                    )
                                else:
                                    st.warning(f"Could not find host coordinates for {ztfid}")
                                    continue
                                
                                # Check for valid coordinates
                                if pd.isna(host_ra) or pd.isna(host_dec):
                                    continue
                                    
                                host_ra_l.append(host_ra)
                                host_dec_l.append(host_dec)
                            except KeyError:
                                st.warning(f"Could not find host data for {ztfid}")
                                continue
                        
                        if not hosts_to_plot:
                            st.warning("No valid hosts found for plotting")
                            return
                        
                        # Create DataFrame exactly like internal code
                        host_df = pd.DataFrame(
                            zip(hosts_to_plot, host_ra_l, host_dec_l),
                            columns=["ztf_object_id", "HOST_RA", "HOST_DEC"],
                        )
                        
                        # Display images in grid
                        if len(host_df) > 0:
                            st.write(f"Loading {len(host_df)} host galaxy images...")
                            progress_bar = st.progress(0)
                            cols = st.columns(3)
                            for i, (_, host) in enumerate(host_df.iterrows()):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    st.markdown(f"**{host['ztf_object_id']}**")
                                    try:
                                        # Try RGB first, fallback to grayscale
                                        img_array = fetch_ps1_rgb_jpeg(host['HOST_RA'], host['HOST_DEC'], size_pix=128)
                                        # Apply stretching like internal plotting code
                                        stretch = AsinhStretch() + PercentileInterval(99.5)
                                        img_stretched = stretch(img_array)
                                        # Convert to PIL Image for display
                                        from PIL import Image
                                        img_pil = Image.fromarray((img_stretched * 255).astype(np.uint8))
                                        st.image(img_pil, use_column_width=True)
                                    except Exception:
                                        try:
                                            img_array = fetch_ps1_cutout(host['HOST_RA'], host['HOST_DEC'], size_pix=128)
                                            # Apply stretching like internal plotting code
                                            stretch = AsinhStretch() + PercentileInterval(99.5)
                                            img_stretched = stretch(img_array)
                                            # Convert to PIL Image for display
                                            from PIL import Image
                                            img_pil = Image.fromarray((img_stretched * 255).astype(np.uint8))
                                            st.image(img_pil, use_column_width=True)
                                        except Exception as e:
                                            st.error(f"Failed to fetch image: {e}")
                                    
                                    # Update progress bar
                                    progress_bar.progress((i + 1) / len(host_df))

                            # Clear progress bar when done
                            progress_bar.empty()
                            st.success(f"Loaded {len(host_df)} host galaxy images!")
                        else:
                            st.warning("No valid host galaxy coordinates found for any neighbors.")
                            
                except Exception as e:
                    st.error(f"Error displaying host galaxy images: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #a0aec0; font-family: \'Inter\', sans-serif;'>
            <p>Built with reLAISS and Streamlit</p>
            <p>For more information, visit the <a href='https://github.com/your-repo/re-laiss' target='_blank' style='color: #4299e1;'>reLAISS repository</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 