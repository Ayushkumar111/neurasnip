"""
NeuraSnip - Semantic Image Search UI
Beautiful Streamlit interface for searching your images
"""

import streamlit as st
from pathlib import Path
import os
from PIL import Image
import tempfile

# Import our modules (clean imports thanks to __init__.py!)
from src import SearchEngine
from src.indexer.image_indexer import ImageIndexer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NeuraSnip - Semantic Image Search",
    page_icon="🔍",
    layout="wide",  # Use full width
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================================================

def load_css():
    """Apply custom styling"""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Search box styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    
    /* Result cards */
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .score-high {
        background: #4CAF50;
        color: white;
    }
    
    .score-medium {
        background: #FF9800;
        color: white;
    }
    
    .score-low {
        background: #9E9E9E;
        color: white;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'search_engine' not in st.session_state:
        with st.spinner("🔄 Loading search engine..."):
            st.session_state.search_engine = SearchEngine(
                db_path="data/vector_store/images.index"
            )
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_score_badge(score: float) -> str:
    """Generate HTML badge for similarity score"""
    percentage = score * 100
    
    if score >= 0.8:
        badge_class = "score-high"
        emoji = "🔥"
    elif score >= 0.6:
        badge_class = "score-medium"
        emoji = "✅"
    else:
        badge_class = "score-low"
        emoji = "⚠️"
    
    return f'<span class="score-badge {badge_class}">{emoji} {percentage:.1f}%</span>'


def display_image_grid(results, cols=3, show_similarity=True):
    """
    Display results in a beautiful grid
    
    Args:
        results: List of result dictionaries
        cols: Number of columns
        show_similarity: Whether to show similarity scores (False for random images)
    """
    if not results:
        st.info("🔍 No results found. Try a different query!")
        return
    
    # Create columns
    columns = st.columns(cols)
    
    for idx, result in enumerate(results):
        col = columns[idx % cols]
        
        with col:
            # Load image
            try:
                img_path = result.get('path', '')
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    
                    # Display image
                    st.image(img, use_container_width=True)
                    
                    # Info container
                    with st.container():
                        # Filename
                        filename = result.get('filename', Path(img_path).name)
                        st.markdown(f"**{filename}**")
                        
                        # Score badge (only if show_similarity is True)
                        if show_similarity:
                            # Try both 'similarity' and 'score' keys for compatibility
                            score = result.get('similarity', result.get('score', 0.0))
                            if score > 0:
                                st.markdown(
                                    get_score_badge(score),
                                    unsafe_allow_html=True
                                )
                        
                        # OCR text preview
                        ocr_text = result.get('ocr_text', '')
                        if ocr_text and ocr_text.strip():
                            with st.expander("📝 OCR Text"):
                                st.text(ocr_text[:200])
                        
                        # Path
                        st.caption(f"📍 {img_path}")
                    
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"Error loading image: {e}")
                import traceback
                st.code(traceback.format_exc())


def display_sidebar():
    """Render sidebar with stats and controls"""
    with st.sidebar:
        st.markdown("## 📊 Statistics")
        
        # Get engine stats
        engine = st.session_state.search_engine
        stats = engine.get_statistics()
        
        # Display stats
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_images']}</div>
            <div class="stat-label">Total Images</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== NEW: DATABASE MANAGEMENT =====
        st.markdown("### 🔄 Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True, help="Scan for new images"):
                with st.spinner("🔄 Scanning for new images..."):
                    try:
                        # Run indexer
                        indexer = ImageIndexer(
                            images_folder="D:/IMAGES",
                            db_path="data/vector_store/images.index",
                            skip_duplicates=True
                        )
                        
                        result = indexer.index_folder()
                        
                        # Reload search engine
                        st.session_state.search_engine = SearchEngine(
                            db_path="data/vector_store/images.index"
                        )
                        
                        # Show results
                        if result['new_indexed'] > 0:
                            st.success(f"✅ Added {result['new_indexed']} new images!")
                        else:
                            st.info("ℹ️ No new images found")
                        
                        # Force refresh
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("📊 Reindex All", use_container_width=True, help="Rebuild entire database"):
                if st.session_state.get('confirm_reindex', False):
                    with st.spinner("🔄 Rebuilding database..."):
                        try:
                            # Delete old index
                            import os
                            if os.path.exists("data/vector_store/images.index"):
                                os.remove("data/vector_store/images.index")
                                os.remove("data/vector_store/images_metadata.pkl")
                            
                            # Reindex everything
                            indexer = ImageIndexer(
                                images_folder="D:/IMAGES",
                                db_path="data/vector_store/images.index",
                                skip_duplicates=False
                            )
                            
                            result = indexer.index_folder()
                            
                            # Reload engine
                            st.session_state.search_engine = SearchEngine(
                                db_path="data/vector_store/images.index"
                            )
                            
                            st.success(f"✅ Reindexed {result['new_indexed']} images!")
                            st.session_state.confirm_reindex = False
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Click again to confirm")
                    st.session_state.confirm_reindex = True
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🤖 Model Info")
        st.info(f"**CLIP Model:** {stats['model_info']['model_name']}")
        st.info(f"**Embedding Dim:** {stats['embedding_dimension']}")
        
        st.markdown("---")
        
        # Search settings
        st.markdown("### ⚙️ Search Settings")
        
        top_k = st.slider(
            "Number of results",
            min_value=5,
            max_value=50,
            value=12,
            step=1
        )
        
        min_similarity = st.slider(
            "Minimum similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05
        )
        
        st.session_state.top_k = top_k
        st.session_state.min_similarity = min_similarity
        
        st.markdown("---")
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📜 Recent Searches")
            for query in st.session_state.search_history[-5:]:
                if st.button(f"🔍 {query}", key=f"history_{query}"):
                    st.session_state.current_query = query
                    st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Load custom CSS
    load_css()
    
    # Initialize
    init_session_state()
    
    # Header
    st.markdown('<h1 class="title">🔍 NeuraSnip</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Semantic Image Search powered by CLIP</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    display_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "🎲 Explore", "📊 Dashboard"])
    
    # ========================================================================
    # TAB 1: SEARCH (WITH IMAGE UPLOAD!)
    # ========================================================================
    with tab1:
        st.markdown("### 🔍 Search Your Images")
        
        # Search mode selector
        search_mode = st.radio(
            "Search by:",
            ["📝 Text Query", "🖼️ Upload Image", "🎯 Hybrid (Text + Image)"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # ====================================================================
        # MODE 1: TEXT SEARCH
        # ====================================================================
        if search_mode == "📝 Text Query":
            st.markdown("#### Enter your search query")
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                query = st.text_input(
                    "Search",
                    placeholder="e.g., 'sunset on beach', 'coffee receipt', 'cat sleeping'",
                    label_visibility="collapsed"
                )
            
            with col2:
                search_button = st.button(
                    "🔍 Search",
                    use_container_width=True,
                    type="primary"
                )
            
            # Perform text search
            if search_button and query:
                with st.spinner("🔄 Searching..."):
                    engine = st.session_state.search_engine
                    
                    results = engine.search_by_text(
                        query=query,
                        top_k=st.session_state.get('top_k', 12),
                        min_similarity=st.session_state.get('min_similarity', 0.3)
                    )
                    
                    formatted_results = engine.format_results(results)
                    st.session_state.current_results = formatted_results
                    
                    if query not in st.session_state.search_history:
                        st.session_state.search_history.append(query)
        
        # ====================================================================
        # MODE 2: IMAGE SEARCH
        # ====================================================================
        elif search_mode == "🖼️ Upload Image":
            st.markdown("#### Upload an image to find similar ones")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose an image...",
                    type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                    label_visibility="collapsed"
                )
            
            with col2:
                if uploaded_file is not None:
                    search_button = st.button(
                        "🔍 Find Similar",
                        use_container_width=True,
                        type="primary"
                    )
                else:
                    st.info("Upload an image first")
                    search_button = False
            
            # Show uploaded image preview
            if uploaded_file is not None:
                st.markdown("##### 👀 Preview:")
                preview_img = Image.open(uploaded_file)
                st.image(preview_img, width=300, caption="Query Image")
            
            # Perform image search
            if uploaded_file is not None and search_button:
                with st.spinner("🔄 Searching for similar images..."):
                    engine = st.session_state.search_engine
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Search by image
                        results = engine.search_by_image(
                            image_path=tmp_path,
                            top_k=st.session_state.get('top_k', 12),
                            min_similarity=st.session_state.get('min_similarity', 0.3)
                        )
                        
                        formatted_results = engine.format_results(results)
                        st.session_state.current_results = formatted_results
                        
                        st.success(f"✅ Found {len(formatted_results)} similar images!")
                        
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
        
        # ====================================================================
        # MODE 3: HYBRID SEARCH
        # ====================================================================
        elif search_mode == "🎯 Hybrid (Text + Image)":
            st.markdown("#### Combine text description + reference image")
            
            # Text input
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query = st.text_input(
                    "Text description",
                    placeholder="e.g., 'red car', 'sunset', 'person smiling'",
                    label_visibility="collapsed"
                )
            
            with col2:
                text_weight = st.slider(
                    "Text importance",
                    0.0, 1.0, 0.7,
                    label_visibility="collapsed"
                )
            
            st.caption(f"Text: {text_weight*100:.0f}% | Image: {(1-text_weight)*100:.0f}%")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload reference image",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
            )
            
            # Preview
            if uploaded_file is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 📝 Text Query:")
                    st.info(query if query else "No text query")
                
                with col2:
                    st.markdown("##### 🖼️ Reference Image:")
                    preview_img = Image.open(uploaded_file)
                    st.image(preview_img, width=200)
            
            # Search button
            search_button = st.button(
                "🎯 Hybrid Search",
                use_container_width=True,
                type="primary",
                disabled=(not query and uploaded_file is None)
            )
            
            # Perform hybrid search
            if search_button:
                if not query and uploaded_file is None:
                    st.error("❌ Please provide either text or image!")
                else:
                    with st.spinner("🔄 Performing hybrid search..."):
                        engine = st.session_state.search_engine
                        
                        # Save temp file if image provided
                        tmp_path = None
                        if uploaded_file is not None:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                        
                        try:
                            # Hybrid search
                            results = engine.search_hybrid(
                                query_text=query if query else None,
                                query_image=tmp_path if tmp_path else None,
                                text_weight=text_weight,
                                image_weight=1 - text_weight,
                                top_k=st.session_state.get('top_k', 12)
                            )
                            
                            formatted_results = engine.format_results(results)
                            st.session_state.current_results = formatted_results
                            
                            st.success(f"✅ Found {len(formatted_results)} results!")
                            
                        finally:
                            # Clean up
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)
        
        # ====================================================================
        # DISPLAY RESULTS (Common for all modes)
        # ====================================================================
        
        st.markdown("---")
        
        # Display results (WITH similarity scores)
        if st.session_state.current_results:
            st.markdown(f"### 📊 Results ({len(st.session_state.current_results)} found)")
            display_image_grid(
                st.session_state.current_results,
                cols=3,
                show_similarity=True  # ✅ Show scores for search results
            )
        else:
            # Show sample queries for text mode
            if search_mode == "📝 Text Query":
                st.markdown("### 💡 Try these sample queries:")
                
                sample_queries = [
                    "sunset on beach",
                    "receipt from coffee shop",
                    "cute cat",
                    "document with text",
                    "food photo",
                    "person smiling"
                ]
                
                cols = st.columns(3)
                for idx, sample in enumerate(sample_queries):
                    col = cols[idx % 3]
                    if col.button(f"🔍 {sample}", key=f"sample_{idx}"):
                        st.session_state.current_query = sample
                        st.rerun()
            
            # Instructions for image mode
            elif search_mode == "🖼️ Upload Image":
                st.info("👆 Upload an image to find similar ones in your collection")
                
                st.markdown("""
                #### 🎯 Use Cases:
                - **Find duplicates** - Upload an image to find all copies
                - **Find variations** - Upload a photo to find similar angles/versions
                - **Style matching** - Upload a style reference to find similar aesthetics
                - **Color matching** - Upload an image to find similar color palettes
                """)
            
            # Instructions for hybrid mode
            elif search_mode == "🎯 Hybrid (Text + Image)":
                st.info("💡 Combine text + image for super precise searches!")
                
                st.markdown("""
                #### 🎯 Example Use Cases:
                - **Text:** "red car" + **Image:** (your car model) → Find red cars of that model
                - **Text:** "sunset" + **Image:** (beach photo) → Find beach sunsets
                - **Text:** "receipt" + **Image:** (Starbucks logo) → Find Starbucks receipts
                - **Text:** "person" + **Image:** (someone's face) → Find photos of that person
                """)
    
    # ========================================================================
    # TAB 2: EXPLORE
    # ========================================================================
    with tab2:
        st.markdown("### 🎲 Discover Random Images")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🎲 Get Random", use_container_width=True, type="primary"):
                engine = st.session_state.search_engine
                random_results = engine.get_random_samples(count=12)
                st.session_state.random_results = random_results
        
        with col2:
            st.info("Click the button to explore random images from your collection")
        
        # Display random results (WITHOUT similarity scores)
        if 'random_results' in st.session_state and st.session_state.random_results:
            st.markdown(f"### 🎨 Random Sample ({len(st.session_state.random_results)} images)")
            display_image_grid(
                st.session_state.random_results,
                cols=4,
                show_similarity=False  # ✅ NO scores for random images
            )
        else:
            st.info("👆 Click 'Get Random' to explore your collection")
    
    # ========================================================================
    # TAB 3: DASHBOARD - remains the same
    # ========================================================================
    with tab3:
        st.markdown("### 📊 Collection Dashboard")
        
        engine = st.session_state.search_engine
        stats = engine.get_statistics()
        
        # Stats cards in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['total_images']}</div>
                <div class="stat-label">📸 Total Images</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['embedding_dimension']}</div>
                <div class="stat-label">🧠 Embedding Dimension</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            searches = len(st.session_state.search_history)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{searches}</div>
                <div class="stat-label">🔍 Searches Performed</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Model details
        st.markdown("### 🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **CLIP Model Details:**
            - Architecture: Vision Transformer
            - Training: Contrastive Learning
            - Capabilities: Text & Image Understanding
            """)
        
        with col2:
            st.markdown(f"""
            **Current Configuration:**
            - Model: {stats['model_info']['model_name']}
            - Dimension: {stats['embedding_dimension']}D
            - Database: FAISS IndexFlatIP
            """)
        
        st.markdown("---")
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📜 Complete Search History")
            for i, query in enumerate(reversed(st.session_state.search_history), 1):
                st.markdown(f"{i}. `{query}`")


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()