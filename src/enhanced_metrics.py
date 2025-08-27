"""
Enhanced Metrics Tab - comprehensive analysis tab
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import numpy as np
from bertopic import BERTopic


def create_enhanced_metrics_tab(df: pd.DataFrame, topic_model: Optional[BERTopic] = None, embeddings: Optional[np.ndarray] = None):
    """
    Create a comprehensive and metrics tab
    This tab includes all the specific metrics mentioned:
    - Number of clusters found: X
    - Number of questions clustered: Y  
    - Number of questions not clustered: Z
    - Clustering Results with detailed breakdown
    - Configuration information
    - Embeddings shape information
    """
    
    st.header("🎯 Comprehensive Analysis Dashboard")
    st.markdown("*Complete clustering results, metrics, and configuration details*")
    
    # Calculate all metrics
    total_questions = len(df)
    clustered_questions = len(df[df['Topic_ID'] != -1])
    unclustered_questions = len(df[df['Topic_ID'] == -1])
    unique_clusters = len(df[df['Topic_ID'] != -1]['Topic_ID'].unique())
    noise_points = unclustered_questions
    noise_percentage = (noise_points / total_questions) * 100
    categorized_percentage = (clustered_questions / total_questions) * 100
    
    # Hero metrics section
    st.subheader("📊 Key Metrics Overview")
    
    # Add a helpful explanation box
    with st.expander("❔ What do these numbers mean?", expanded=False):
        st.markdown("""
        **🎯 Total Questions:** How many questions we analyzed (like counting all students in a school)
        
        **🎪 Clusters Found:** How many different topic groups we discovered (like different clubs or subjects)
        
        **✅ Questions Clustered:** Questions we successfully put into topic groups (like students who joined clubs)
        
        **❌ Questions Not Clustered:** Questions that didn't fit any topic clearly (like students who haven't found their group yet)
        
        **📈 Categorization Rate:** What percentage of questions we successfully organized (higher is better!)
        """)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 Total Questions",
            value=total_questions,
            help="Total number of questions analyzed - this is our complete dataset size"
        )
    
    with col2:
        st.metric(
            label="🎪 Clusters Found",
            value=unique_clusters,
            help="Number of distinct topic clusters discovered - each cluster represents a different type of question students ask"
        )
    
    with col3:
        st.metric(
            label="✅ Questions Clustered", 
            value=clustered_questions,
            help="Questions successfully assigned to clusters - these questions have clear topics the AI could identify"
        )
    
    with col4:
        st.metric(
            label="❌ Questions Not Clustered",
            value=unclustered_questions,
            help="Questions that couldn't be categorized into any topic - these might be unique, unclear, or need more examples to form a topic"
        )
    
    with col5:
        st.metric(
            label="📈 Categorization Rate",
            value=f"{categorized_percentage:.1f}%",
            help="Percentage of questions successfully categorized - shows how well the AI understood your question patterns"
        )
    
    st.divider()
    
    # Visual breakdown
    st.subheader("📊 Visual Analytics")
    
    # Add explanation for the visual section
    st.info("👀 **Visual Guide:** These charts help you see your data at a glance. The pie chart shows the big picture of categorized vs uncategorized questions, while the bar chart shows which topics are most common.")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**🥧 Categorization Breakdown**")
        # Categorization pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Categorized', 'Uncategorized'],
            values=[clustered_questions, unclustered_questions],
            hole=0.4,
            marker_colors=['#2E8B57', '#FF6B6B']
        )])
        
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='%{label}<br>%{value} questions<br>%{percent}<extra></extra>'
        )
        
        fig_pie.update_layout(
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, key="categorization_pie_chart")
    
    with viz_col2:
        st.markdown("**📊 Top 10 Most Common Topics**")
        # Top topics bar chart
        topic_counts = df[df['Topic_ID'] != -1].groupby('Topic_Name').size().reset_index(name='Count')
        topic_counts = topic_counts.sort_values('Count', ascending=False).head(10)
        
        fig_bar = px.bar(
            topic_counts,
            x='Count',
            y='Topic_Name',
            orientation='h',
            title="Top 10 Topics by Question Count",
            color='Count',
            color_continuous_scale='viridis'
        )
        
        fig_bar.update_layout(
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, key="top_topics_bar_chart")
        
        with st.expander("❔ How to interpret this chart"):
            st.markdown("""
            **📊 Reading the Bar Chart:**
            - **Longer bars** = More students ask about this topic
            - **Shorter bars** = Less common questions  
            - **Order matters** = Topics are sorted from most to least common
            - **Colors** = Just for visual appeal, darker usually means higher values
            
            **💡 What this tells you:**
            - Which topics need the most attention from support staff
            - What questions to prepare FAQ answers for
            - Where to focus training and resources
            """)
    
    # Success indicators
    st.subheader("🎉 Analysis Quality Indicators")
    
    # Add explanation for quality indicators
    with st.expander("❔ What do these quality indicators mean?"):
        st.markdown("""
        **🎯 Categorization Quality:** How well the AI organized your questions
        - 🟢 **Excellent (80%+):** AI understood most question patterns clearly
        - 🟡 **Good (60-79%):** AI did well, some questions were unclear  
        - 🔴 **Needs Improvement (<60%):** Many questions were too unique or unclear
        
        **⚖️ Cluster Distribution:** Whether we have the right number of topic groups
        - 🟢 **Well Balanced (10-50 clusters):** Good variety without being overwhelming
        - 🟡 **Moderate (5-9 clusters):** Okay but might be too broad
        - 🔴 **Too Few (<5 clusters):** Topics are probably too general
        
        **⚙️ Configuration Status:** Technical settings that affect results
        - Shows if the AI is using the best settings for your data
        """)
    
    # Create a clean table format for quality indicators
    categorization_quality = "🟢 Excellent" if categorized_percentage > 80 else "🟡 Good" if categorized_percentage > 60 else "🔴 Needs Improvement"
    cluster_balance = "🟢 Well Balanced" if unique_clusters >= 10 and unique_clusters <= 50 else "🟡 Moderate" if unique_clusters >= 5 else "🔴 Too Few Clusters"
    min_cluster_quality = "🟢 Optimized" if unique_clusters > 0 else "🔴 No Clusters"
    
    # Create quality indicators dataframe for clean display
    quality_data = {
        "Quality Metric": [
            "🎯 Categorization Quality",
            "⚖️ Cluster Distribution", 
            "⚙️ Configuration Status"
        ],
        "Status": [
            categorization_quality,
            cluster_balance,
            min_cluster_quality
        ],
        "Details": [
            f"{categorized_percentage:.1f}% categorized • {unique_clusters} topics discovered",
            f"{unique_clusters} clusters • Avg {total_questions / max(unique_clusters, 1):.1f} questions/cluster",
            "Min cluster: 15 • Latest OpenAI models • Enhanced labeling"
        ]
    }
    
    quality_df = pd.DataFrame(quality_data)
    
    # Display as a clean table
    st.dataframe(
        quality_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Quality Metric": st.column_config.TextColumn("Quality Metric", width=180),
            "Status": st.column_config.TextColumn("Status", width=150),
            "Details": st.column_config.TextColumn("Details", width=350)
        }
    )
    
    # Configuration details table
    st.markdown("### ⚙️ Technical Configuration")
    
    # Add explanation for configuration
    with st.expander("❔ What do these technical settings mean?"):
        st.markdown("""
        **🤖 AI Models:** The specific OpenAI models used for understanding and generating text
        - **Embedding Model:** Converts questions into numbers the AI can understand
        - **Chat Model:** Generates the topic names and descriptions you see
        
        **🎯 Clustering Settings:** How the AI groups similar questions together
        - **Min Cluster Size:** Minimum questions needed to form a topic group
        - **UMAP Neighbors:** How many nearby questions to consider when grouping
        - **Dimensions:** How many aspects the AI considers when comparing questions
        
        **📊 Other Settings:** Additional technical parameters that affect the analysis
        """)
    
    # Import config values
    from config import EMBEDDING_MODEL, CHAT_MODEL, MIN_CLUSTER_SIZE, UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, MAX_FEATURES
    
    # Get embeddings shape information if available
    if embeddings is not None and hasattr(embeddings, 'shape'):
        embeddings_shape = f"{embeddings.shape[0]} questions × {embeddings.shape[1]} dimensions"
    elif embeddings is not None and isinstance(embeddings, dict):
        embeddings_shape = f"Dictionary with {len(embeddings)} entries"
    else:
        embeddings_shape = "Not available"
    
    # Create configuration table
    config_data = {
        "Configuration": [
            "🤖 OpenAI Embedding Model",
            "🤖 OpenAI Chat Model", 
            "🎯 Min Cluster Size",
            "🗺️ UMAP Neighbors",
            "📐 UMAP Dimensions",
            "📊 Max Features",
            "🧮 Embeddings Shape"
        ],
        "Value": [
            EMBEDDING_MODEL,
            CHAT_MODEL,
            str(MIN_CLUSTER_SIZE),
            str(UMAP_N_NEIGHBORS), 
            str(UMAP_N_COMPONENTS),
            str(MAX_FEATURES),
            embeddings_shape
        ],
        "Description": [
            "AI model that converts text to numbers",
            "AI model that generates topic names",
            "Minimum questions needed per topic",
            "How many nearby questions to consider",
            "Number of dimensions for clustering",
            "Maximum vocabulary size for analysis",
            "Size of the numerical representation"
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    
    # Display configuration table
    st.dataframe(
        config_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Configuration": st.column_config.TextColumn("Configuration", width=200),
            "Value": st.column_config.TextColumn("Value", width=150),
            "Description": st.column_config.TextColumn("Description", width=300)
        }
    )
    
    # Download section for review format
    st.divider()
    st.subheader("📥 Download for Review")
    
    # Add explanation for downloads
    st.info("💾 **Download Options:** Get your analysis results in different formats for different purposes!")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        st.markdown("**📋 Complete Data Download**")
        st.caption("Perfect for further analysis or sharing with technical team")
        csv_standard = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Complete Analysis",
            data=csv_standard,
            file_name=f"pathway_questions_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Full analysis with all columns and metrics - includes confidence scores, topic IDs, and all technical details"
        )
    
    with download_col2:
        st.markdown("**📝 Elder Edwards Clean Review Format**")
        st.caption("Simplified format for easy reading and review")
        # Create the exact format: representation and question columns, sorted
        review_df = df[['Topic_Name', 'Question']].copy()
        review_df = review_df.rename(columns={'Topic_Name': 'representation'})
        # Sort by representation (topic) and then by question alphabetically
        review_df = review_df.sort_values(['representation', 'Question'])
        
        csv_review = review_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Review Format", 
            data=csv_review,
            file_name=f"pathway_questions_review_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Clean format with just topic names and questions, sorted by topic for easy review - perfect for managers and reviewers"
        )
    
    # Topic model insights if available
    if topic_model is not None:
        st.divider()
        st.subheader("🔬 Advanced Topic Analysis")
        
        try:
            # Topic info
            topic_info = topic_model.get_topic_info()
            
            if len(topic_info) > 1:  # More than just the noise topic
                st.markdown("**📊 Topic Details**")
                
                # Display topic information table
                display_topics = topic_info[topic_info['Topic'] != -1].head(10)
                st.dataframe(
                    display_topics[['Topic', 'Count', 'Name']],
                    use_container_width=True,
                    column_config={
                        'Topic': st.column_config.NumberColumn('Topic ID', width=100),
                        'Count': st.column_config.NumberColumn('Questions', width=100),
                        'Name': st.column_config.TextColumn('Topic Description', width=300)
                    }
                )
                
                # Topic keywords
                with st.expander("🔤 View Topic Keywords", expanded=False):
                    for topic_id in display_topics['Topic'][:5]:  # Show top 5 topics
                        if topic_id != -1:
                            keywords = topic_model.get_topic(topic_id)[:8]
                            keyword_str = ", ".join([f"**{word}** ({score:.3f})" for word, score in keywords])
                            st.markdown(f"**Topic {topic_id}:** {keyword_str}")
        
        except Exception as e:
            st.info(f"Advanced topic analysis not available: {str(e)}")
    
    # Success message and next steps
    st.success("""
    🎉 **Analysis Complete!** 
    
    Your questions have been successfully analyzed and clustered. The results show **{:.1f}% categorization rate** 
    with **{} distinct topics** identified. This analysis can now be used for:
    
    • Understanding student question patterns
    • Identifying common areas of confusion  
    • Improving support resources
    • Training chatbot responses
    """.format(categorized_percentage, unique_clusters))
    
    return {
        'total_questions': total_questions,
        'clusters_found': unique_clusters,
        'questions_clustered': clustered_questions,
        'questions_not_clustered': unclustered_questions,
        'categorized_percentage': categorized_percentage,
        'noise_percentage': noise_percentage
    }
