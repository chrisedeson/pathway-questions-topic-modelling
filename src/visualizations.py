"""
Visualization components for BYU Pathway Questions Analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


def display_interactive_scatter(df: pd.DataFrame, topic_model: Optional[BERTopic] = None, embeddings: Optional[np.ndarray] = None):
    """Display interactive scatter plot of questions"""
    st.subheader("🎯 Question Distribution by Topic")
    
    if topic_model is not None and embeddings is not None:
        try:
            # Create 2D UMAP embeddings for visualization
            import umap.umap_ as umap
            umap_viz = umap.UMAP(
                n_neighbors=15, 
                n_components=2, 
                random_state=42, 
                metric='cosine'
            )
            viz_embeddings = umap_viz.fit_transform(embeddings)
            
            # Create scatter plot with unique key
            fig = px.scatter(
                x=viz_embeddings[:, 0],
                y=viz_embeddings[:, 1],
                color=df['Topic_Name'],
                hover_data={'Question': df['Question'], 'Confidence': df['Probability'].round(3)},
                title="Questions Clustered by Topic",
                labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
                width=800,
                height=600
            )
            
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>" +
                              "Topic: %{fullData.name}<br>" +
                              "Confidence: %{customdata[1]}<br>" +
                              "<extra></extra>",
                customdata=list(zip(df['Question'], df['Probability'].round(3)))
            )
            
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, key="scatter_plot_interactive")
            st.info("💡 **Tip:** Hover over points to see individual questions! Points closer together represent similar questions.")
            return
                
        except Exception as e:
            st.warning(f"Could not create interactive scatter plot: {str(e)}")
    
    # Fallback visualization with unique key
    display_topic_distribution_chart(df, chart_key="topic_distribution_fallback")


def display_topic_distribution_chart(df: pd.DataFrame, chart_key: str = "topic_distribution_bar"):
    """Display topic distribution bar chart with unique key"""
    topic_counts = df.groupby('Topic_Name').size().reset_index(name='Count')
    topic_counts = topic_counts.sort_values('Count', ascending=False)
    
    fig = px.bar(
        topic_counts, 
        x='Topic_Name', 
        y='Count',
        title="Questions Distribution by Topic",
        labels={'Count': 'Number of Questions', 'Topic_Name': 'Topic'},
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_tickangle=45,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def display_topic_hierarchy(topic_model: BERTopic):
    """Display topic hierarchy visualization"""
    st.subheader("🌳 Topic Hierarchy & Clustering")
    
    try:
        # Hierarchical clustering visualization
        fig_hier = topic_model.visualize_hierarchy()
        fig_hier.update_layout(height=700)
        st.plotly_chart(fig_hier, use_container_width=True, key="topic_hierarchy")
        
        st.info("💡 **Tip:** This dendrogram shows how topics relate hierarchically. Similar topics cluster together.")
        
    except Exception as e:
        st.error(f"Could not create hierarchy visualization: {str(e)}")


def display_topic_similarity_heatmap(topic_model: BERTopic):
    """Display topic similarity heatmap"""
    try:
        if hasattr(topic_model, 'topic_embeddings_') and topic_model.topic_embeddings_ is not None:
            st.subheader("🔥 Topic Similarity Heatmap")
            
            # Calculate cosine similarity between topics
            topic_embeddings = topic_model.topic_embeddings_
            similarity_matrix = cosine_similarity(topic_embeddings)
            
            # Get topic names
            topic_info = topic_model.get_topic_info()
            topic_names = [f"Topic {i}: {name[:30]}..." if len(name) > 30 else f"Topic {i}: {name}" 
                         for i, name in enumerate(topic_info['Name'][:20])]  # Limit to top 20
            
            if len(similarity_matrix) > 20:
                similarity_matrix = similarity_matrix[:20, :20]
            
            fig_heatmap = px.imshow(
                similarity_matrix,
                labels=dict(x="Topics", y="Topics", color="Similarity"),
                x=topic_names,
                y=topic_names,
                title="Topic Similarity Matrix",
                color_continuous_scale='RdYlBu_r',
                aspect="auto"
            )
            fig_heatmap.update_layout(
                height=600,
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True, key="topic_similarity_heatmap")
            st.info("💡 **Tip:** Darker red indicates higher similarity between topics.")
            
    except Exception as e:
        st.warning(f"Could not create similarity heatmap: {str(e)}")


def display_confidence_distribution(df: pd.DataFrame):
    """Display confidence/probability distribution"""
    st.subheader("📊 Confidence Distribution")
    
    # Histogram of confidence scores
    fig_hist = px.histogram(
        df, 
        x='Probability', 
        nbins=30,
        title="Distribution of Topic Assignment Confidence",
        labels={'Probability': 'Confidence Score', 'count': 'Number of Questions'},
        color_discrete_sequence=['#1f77b4']
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True, key="confidence_histogram")
    
    # Box plot by topic
    fig_box = px.box(
        df[df['Topic_ID'] != -1], 
        x='Topic_Name', 
        y='Probability',
        title="Confidence Distribution by Topic",
        labels={'Probability': 'Confidence Score', 'Topic_Name': 'Topic'}
    )
    fig_box.update_layout(
        xaxis_tickangle=45,
        height=500
    )
    st.plotly_chart(fig_box, use_container_width=True, key="confidence_box_plot")


def display_topic_words_chart(topic_model: BERTopic):
    """Display top words for each topic"""
    st.subheader("🔤 Top Words by Topic")
    
    try:
        topic_info = topic_model.get_topic_info()
        
        # Select topics to display (excluding noise topic -1)
        topics_to_show = topic_info[topic_info['Topic'] != -1]['Topic'].head(10).tolist()
        
        # Create word importance data
        word_data = []
        for topic_id in topics_to_show:
            words = topic_model.get_topic(topic_id)[:8]  # Top 8 words
            for word, score in words:
                word_data.append({
                    'Topic': f"Topic {topic_id}",
                    'Word': word,
                    'Score': score
                })
        
        if word_data:
            word_df = pd.DataFrame(word_data)
            
            fig = px.bar(
                word_df, 
                y='Topic', 
                x='Score', 
                color='Word',
                orientation='h',
                title="Top Words by Topic (TF-IDF Scores)",
                height=600
            )
            
            fig.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True, key="topic_words_chart")
            
    except Exception as e:
        st.warning(f"Could not create topic words chart: {str(e)}")


def display_visualization_tabs(df: pd.DataFrame, topic_model: Optional[BERTopic] = None, embeddings: Optional[np.ndarray] = None):
    """Display all visualizations in organized tabs"""
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "🎯 Question Scatter", 
        "🌳 Topic Hierarchy", 
        "📊 Distributions", 
        "🔤 Topic Words"
    ])
    
    with viz_tab1:
        display_interactive_scatter(df, topic_model, embeddings)
    
    with viz_tab2:
        if topic_model is not None:
            display_topic_hierarchy(topic_model)
            display_topic_similarity_heatmap(topic_model)
        else:
            st.info("Topic model not available for hierarchy visualization.")
    
    with viz_tab3:
        display_confidence_distribution(df)
        display_topic_distribution_chart(df, chart_key="topic_distribution_main")
    
    with viz_tab4:
        if topic_model is not None:
            display_topic_words_chart(topic_model)
        else:
            st.info("Topic model not available for word analysis.")


def display_metrics_overview(metrics: dict):
    """Display comprehensive metrics as requested in tasks"""
    st.subheader("📈 Comprehensive Analysis Metrics")
    
    # Main metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Questions", metrics['total_questions'])
    
    with col2:
        st.metric("Clusters Found", metrics['clusters_found'])
    
    with col3:
        st.metric("Questions Clustered", metrics['questions_clustered'])
    
    with col4:
        st.metric("Not Clustered", metrics['questions_not_clustered'])
    
    with col5:
        st.metric("Categorized", f"{metrics['categorized_percentage']:.1f}%")
    
    # Detailed metrics in expandable section
    with st.expander("📊 Detailed Clustering Results", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Clustering Results:**
            - Clusters found: **{metrics['clusters_found']}**
            - Noise points: **{metrics['noise_points']} ({metrics['noise_percentage']:.1f}%)**
            - Questions categorized: **{metrics['categorized_percentage']:.1f}%**
            - Min Cluster Size: **{metrics['min_cluster_size']}**
            """)
        
        with col2:
            st.markdown(f"""
            **Configuration:**
            - Embedding Model: **text-embedding-3-large**
            - Chat Model: **gpt-4o-mini**
            - Questions analyzed: **{metrics['total_questions']}**
            """)
        
        if 'embeddings_shape' in metrics:
            st.markdown(f"""
            **Embeddings Information:**
            - Dimensions: **{metrics['embeddings_shape']}**
            """)


def create_download_section(df: pd.DataFrame):
    """Create download section with requested CSV format"""
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Standard Download**")
        csv_standard = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Full Analysis CSV",
            data=csv_standard,
            file_name=f"pathway_questions_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download complete analysis with all columns"
        )
    
    with col2:
        st.markdown("**📝 Elder Edwards Review Format**")
        # Create the specific format requested in tasks: representation and question columns
        review_df = df[['Topic_Name', 'Question']].copy()
        review_df = review_df.rename(columns={'Topic_Name': 'representation'})
        # Sort by representation and then by question alphabetically as requested
        review_df = review_df.sort_values(['representation', 'Question'])
        
        csv_review = review_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Review CSV",
            data=csv_review,
            file_name=f"pathway_questions_review_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download sorted by topic (representation) and question for easy review"
        )
