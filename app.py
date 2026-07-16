import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & DUNE THEME INJECTION
# ==========================================
st.set_page_config(
    page_title="TechNexus | Arrakis Curation Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the Dune (Arrakis) Basalt & Spice Aesthetic
st.markdown(
    """
    <style>
    /* Global Background and Typography Overrides */
    .stApp {
        background-color: #080605; /* Deep Canyon Obsidian */
    }
    
    /* Force the entire ARRAKIS CONSOLE sidebar background to solid pitch black */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #c5a059;
    }
    
    /* Force all textual input elements and labels inside the sidebar to pure white */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Brutalist Basalt Cards with Shimmering Spice Dust Aura */
    .metric-card {
        background-color: #120e0c; /* Basalt Rock Shelter */
        border: 1px solid #c5a059; /* Dull Spice Gold Rim */
        border-radius: 6px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(197, 160, 89, 0.15); /* Warm Spice Glow */
        margin-bottom: 15px;
    }
    
    /* Specific Glowing Accents */
    .spice-text {
        color: #ffffff !important; 
        font-family: 'Courier New', monospace;
        font-weight: bold;
        letter-spacing: 2px;
    }
    
    .ibad-text {
        color: #00d2ff; /* Deep Melange Blue */
        font-family: 'Courier New', monospace;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    /* Override standard Streamlit headers */
    h1, h2, h3, h4 {
        color: #f4eae1 !important; /* Sun-bleached Sandstone */
        font-family: 'Georgia', serif;
    }
    
    p, span, label {
        color: #c4b5a9 !important; /* Soft desert dust */
    }

    /* ==========================================================
       FIXED: FORCE ALL POP-OVER DEPLOY OPTIONS & DESCRIPTIONS TO BLACK
       ========================================================== */
    [data-testid="stPopoverBody"] *, 
    [data-testid="stPopoverBody"] p,
    [data-testid="stPopoverBody"] span,
    [data-testid="stPopoverBody"] div,
    [data-testid="stPopoverBody"] h1,
    [data-testid="stPopoverBody"] h2,
    [data-testid="stPopoverBody"] h3,
    [data-testid="stPopoverBody"] li,
    [data-testid="stPopover"] div, 
    [data-testid="stPopover"] p, 
    [data-testid="stPopover"] li, 
    [data-testid="stPopover"] span,
    div[data-testid="stConnectionStatus"] p,
    div[data-testid="stConnectionStatus"] span,
    div[data-testid="stConnectionStatus"] li {
        color: #000000 !important;
        font-weight: 500 !important;
    }

    /* Force the Deploy Action Button text to Pure White */
    .stDeployButton button, [data-testid="stDeployButton"] {
        color: #ffffff !important;
    }

    /* Target the built-in Streamlit footer section to make background Black and text White */
    footer {
        background-color: #000000 !important;
        border-top: 1px solid #c5a059;
        opacity: 1 !important;
        visibility: visible !important;
    }
    footer p, footer a, footer div {
        color: #ffffff !important;
        font-weight: bold !important;
        opacity: 1 !important;
    }

    /* Target the app settings, menu action links, widgets and system controls */
    div[role="menu"], ul[role="listbox"], [data-testid="stStatusWidget"] {
        background-color: #000000 !important;
        border: 1px solid #c5a059 !important;
    }
    div[role="menuitem"], div[role="menuitem"] span, 
    button[data-testid="baseButton-header"], 
    button[data-testid="baseButton-headerNoPadding"],
    .stActionButton button {
        color: #ffffff !important;
    }

    /* Raw Sieve Analytics table text overrides to Black & Central Layout alignment */
    [data-testid="stDataFrame"] th, 
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] .row_heading, 
    [data-testid="stDataFrame"] .col_heading,
    .dataframe th, 
    .dataframe td {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    [data-testid="stDataFrame"] th, 
    .dataframe th,
    [data-testid="stDataFrame"] div[data-testid="stTable"] th {
        text-align: center !important;
    }
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border-radius: 6px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# ADVANCED MATHEMATICAL RECOMMENDER CLASS
# ==========================================
class ItemBasedRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity_df = None

    def fit(self, interaction_df: pd.DataFrame):
        self.user_item_matrix = interaction_df.pivot_table(
            index="User", columns="Item", values="Rating", fill_value=0
        )
        similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

    def recommend(self, target_item: str, top_n: int = 2) -> pd.Series:
        if self.item_similarity_df is None:
            raise ValueError("Model is not trained.")
        similarity_scores = self.item_similarity_df[target_item]
        sorted_scores = similarity_scores.sort_values(ascending=False)
        return sorted_scores.iloc[1 : top_n + 1]

    def get_spatial_embeddings(self) -> pd.DataFrame:
        item_features = self.user_item_matrix.T
        pca = PCA(n_components=2, random_state=42)
        reduced_coordinates = pca.fit_transform(item_features)
        coords_df = pd.DataFrame(reduced_coordinates, columns=["Dimension 1", "Dimension 2"])
        coords_df["Item"] = item_features.index
        return coords_df

# ==========================================
# DATASET GENERATION
# ==========================================
complex_telemetry_data = {
    "User": ["User_A", "User_A", "User_A", "User_A", "User_B", "User_B", "User_B", "User_C", "User_C", "User_C", "User_D", "User_D", "User_E", "User_E", "User_F", "User_F"],
    "Item": ["Post_A", "Post_B", "Post_D", "Post_E", "Post_B", "Post_C", "Post_F", "Post_A", "Post_C", "Post_D", "Post_D", "Post_E", "Post_A", "Post_F", "Post_C", "Post_E"],
    "Rating": [5, 4, 3, 2, 5, 2, 4, 4, 3, 5, 4, 5, 5, 1, 4, 3]
}
df = pd.DataFrame(complex_telemetry_data)

engine = ItemBasedRecommender()
engine.fit(df)

# ==========================================
# SIDEBAR CONTROL CONFIGURATION
# ==========================================
st.sidebar.markdown("## <span class='spice-text'>ARRAKIS CONSOLE</span>", unsafe_allow_html=True)
st.sidebar.write("Calibrate recommendation vectors:")
selected_post = st.sidebar.selectbox("Select Seed Asset:", df["Item"].unique())
num_recommendations = st.sidebar.slider("Spice Flow Depth (Top N):", 1, 5, 3)
enable_vector_links = st.sidebar.checkbox("Project Vector Connections", value=True)

st.title("🏜️ TechNexus: Arrakis Spice-Flow Curation Engine")
st.markdown("*Deconstructing high-dimensional neural filters and cognitive validation loops in cloud-assisted systems.*")
st.markdown("---")

# Row 1: Basalt Dashboard Cards
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="letter-spacing: 1px; color: #a89485;">📡 INTENSITY OF THE SPICE BLOW</h4>
            <h2 class="spice-text" style="color: #e5b869 !important;">ACTIVE & STREAMING</h2>
            <p style="font-size: 0.85em; color: #8a786b; margin: 0;">Ingesting Live Ubuntu Server Packets</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_b:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="letter-spacing: 1px; color: #a89485;">📐 BASALT VECTOR MATRIX</h4>
            <h2 class="spice-text" style="color: #e5b869 !important;">{df['User'].nunique()} Users × {df['Item'].nunique()} Items</h2>
            <p style="font-size: 0.85em; color: #8a786b; margin: 0;">Multi-Dimensional Sieve Shape</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_c:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="letter-spacing: 1px; color: #a89485;">👁️ INFRASTRUCTURE VELOCITY</h4>
            <h2 class="ibad-text">0.024 ms</h2>
            <p style="font-size: 0.85em; color: #8a786b; margin: 0;">Spatial Projection Compute Delay</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# ROW 2: SYSTEM VISUALIZERS & FLOW DIAGRAMS
# ==========================================
st.markdown("## 🔮 System Visualizers & Flow Diagrams")

exp_tabs = st.tabs(["🚀 Ingestion Pipeline Diagram", "⏳ 4-Stage Filtration Funnel", "📊 Surface Signal Weight Matrices"])

with exp_tabs[0]:
    st.subheader("Architectural Flow Diagram: Payload Ingestion & Decoupling")
    st.write("Tracing the journey of a media payload from client-side upload to edge CDN caching.")
    
    st.graphviz_chart(
        """
        digraph G {
            background="#120e0c";
            node [style=filled, color="#c5a059", fillcolor="#211a17", fontcolor="#f4eae1", fontname="Georgia", shape=box];
            edge [color="#e5b869", fontcolor="#c4b5a9", fontname="Georgia"];
            
            Payload [label="User Media Payload\\n(POST /media)"];
            Gateway [label="Gateway & Anti-DDoS Layer"];
            Balancer [label="Load Balancer"];
            Server [label="Application Server\\n(Decoupling Process)"];
            
            ObjStore [label="Object Storage (Ceph/S3)\\n[Large Media]"];
            MetaDB [label="Metadata Database\\n[Transactional Data]"];
            
            CDN [label="Edge CDNs\\n(cdninstagram.com)"];
            Funnel [label="4-Stage Ranking Funnel"];
            
            Payload -> Gateway;
            Gateway -> Balancer;
            Balancer -> Server;
            
            Server -> ObjStore [label="Decoupled Binaries", fontcolor="#000000"];
            Server -> MetaDB [label="Decoupled Metadata", fontcolor="#000000"];
            ObjStore -> CDN [label="Global Caching", fontcolor="#000000"];
            MetaDB -> Funnel [label="Signals Feed", fontcolor="#000000"];
            CDN -> Funnel [label="Instant Playback", fontcolor="#000000"];
        }
        """
    )

with exp_tabs[1]:
    st.subheader("Process Funnel: Multi-Stage Filtration Cascade")
    st.write("Computational cost-optimization pathway transforming billions of candidate posts down to localized recommendations.")
    
    funnel_stages = [
        "1. Universe of Content (Billions)",
        "2. Candidate Retrieval (Two Tower Similarity Search)",
        "3. Early-Stage Filtering (ESR Fast Proxy)",
        "4. Late-Stage Ranking (MTML Neural Net)",
        "5. Final Polish (Integrity & Business Rules)"
    ]
    funnel_values = [1000000, 10000, 500, 100, 20]
    
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_stages,
        x=funnel_values,
        textinfo="value+percent initial",
        textfont=dict(color="#FFFFFF", size=14, family="Georgia"),
        marker={"color": ["#4e413c", "#705d54", "#977e70", "#c5a059", "#e5b869"]}
    ))
    
    fig_funnel.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia", color="#f4eae1")
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

with exp_tabs[2]:
    st.subheader("Pictorial Graph: Signal Matrix Tuning Map")
    st.write("Comparative layout showing how relative mathematical weights shift across distinct surface recommendation targets.")
    
    surfaces = ["Feed", "Stories", "Explore", "Reels"]
    
    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=surfaces, y=[95, 90, 10, 5],
        name="User-Author Closeness (Relationships)",
        marker_color="#e5b869"
    ))
    fig_signals.add_trace(go.Bar(
        x=surfaces, y=[40, 20, 85, 95],
        name="User Behavioral Activity (Dwell/Watch Time)",
        marker_color="#00d2ff"
    ))
    fig_signals.add_trace(go.Bar(
        x=surfaces, y=[60, 30, 90, 80],
        name="Asset Virality (Global Velocity of Popularity)",
        marker_color="#a89485"
    ))
    
    fig_signals.update_layout(
        barmode="group",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia", color="#f4eae1"),
        yaxis=dict(title="Calculated Engine Weight Allocation (%)", gridcolor="#211a17"),
        legend=dict(font=dict(color="#FFFFFF", family="Georgia", size=12))
    )
    st.plotly_chart(fig_signals, use_container_width=True)

st.markdown("---")

# Row 3: Live Curation Interface Layout
main_col1, main_col2 = st.columns([3, 2])

# Left side content column (PCA Projection & Raw Data Panels)
with main_col1:
    st.subheader("🗺️ Live Vector Space Projection")
    st.write("Interactively view the PCA projection mapping spatial clusters in sandstone coordinate space.")
    
    coords = engine.get_spatial_embeddings()
    recs = engine.recommend(selected_post, top_n=num_recommendations)

    def check_status(item):
        if item == selected_post:
            return "Active Seed (Target)"
        elif item in recs.index:
            return "Melange Connection (Match)"
        return "Unmapped Basalt Dust"

    coords["Class"] = coords["Item"].apply(check_status)

    fig = px.scatter(
        coords,
        x="Dimension 1",
        y="Dimension 2",
        text="Item",
        color="Class",
        color_discrete_map={
            "Active Seed (Target)": "#e5b869",
            "Melange Connection (Match)": "#00d2ff",
            "Unmapped Basalt Dust": "#4e413c"
        },
        size=[30 if c != "Unmapped Basalt Dust" else 15 for c in coords["Class"]],
        template="plotly_dark"
    )

    fig.update_traces(textposition="top center", hoverinfo="all")
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#211a17"),
        yaxis=dict(showgrid=True, gridcolor="#211a17"),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(color="#FFFFFF", size=12, family="Georgia")
        )
    )

    if enable_vector_links:
        seed_coords = coords[coords["Item"] == selected_post].iloc[0]
        for item in recs.index:
            rec_coords = coords[coords["Item"] == item].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[seed_coords["Dimension 1"], rec_coords["Dimension 1"]],
                    y=[seed_coords["Dimension 2"], rec_coords["Dimension 2"]],
                    mode="lines",
                    line=dict(color="#00d2ff", width=2, dash="dot"),
                    name=f"Spice Hook ({item})",
                    showlegend=False
                )
            )

    st.plotly_chart(fig, use_container_width=True)

    # Raw Sieve Analytical Tables placed directly below the vector space panel
    st.markdown("---")
    st.subheader("📜 Raw Sieve Analytics")
    st.write("Full high-dimensional dataset matrices scaled across the system architecture.")

    matrix_tabs = st.tabs(["Space Ingestion Grids", "Cosine Commutatives"])

    with matrix_tabs[0]:
        st.dataframe(
            engine.user_item_matrix.style.highlight_max(
                axis=1, 
                props="background-color: #000000; color: #ffffff; font-weight: bold; border: 1.5px solid #e5b869;"
            ),
            use_container_width=True
        )

    with matrix_tabs[1]:
        st.dataframe(
            engine.item_similarity_df.style.background_gradient(cmap="YlOrBr"),
            use_container_width=True
        )

# Right side column (Melange Extraction Rankings Only)
with main_col2:
    st.subheader("🏺 Melange Extraction Rankings")
    st.write("Proximity scoring based on computed cosine vector angles:")

    for rank, (item_id, score) in enumerate(recs.items(), 1):
        st.markdown(
            f"""
            <div style="background-color: #120e0c; padding: 14px; border-left: 4px solid #e5b869; margin-bottom: 8px; border-radius: 4px;">
                <span style="font-weight: bold; color: #f4eae1;">Rank {rank}: {item_id}</span><br/>
                <span style="font-size: 0.9em; color: #00d2ff;">Cosine Alignment Strength: {score:.4f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )