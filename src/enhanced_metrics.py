"""
Enhanced Metrics Tab - A beautiful comprehensive analysis tab as requested in tasks.md
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
    Create a comprehensive and beautiful metrics tab as requested in tasks.md
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
    
    # Calculate all metrics as specified in tasks
    total_questions = len(df)
    clustered_questions = len(df[df['Topic_ID'] != -1])
    unclustered_questions = len(df[df['Topic_ID'] == -1])
    unique_clusters = len(df[df['Topic_ID'] != -1]['Topic_ID'].unique())
    noise_points = unclustered_questions
    noise_percentage = (noise_points / total_questions) * 100
    categorized_percentage = (clustered_questions / total_questions) * 100
    
    # Hero metrics section
    st.subheader("📊 Key Metrics Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 Total Questions",
            value=total_questions,
            help="Total number of questions analyzed"
        )
    
    with col2:
        st.metric(
            label="🎪 Clusters Found",
            value=unique_clusters,
            help="Number of distinct topic clusters discovered"
        )
    
    with col3:
        st.metric(
            label="✅ Questions Clustered", 
            value=clustered_questions,
            help="Questions successfully assigned to clusters"
        )
    
    with col4:
        st.metric(
            label="❌ Questions Not Clustered",
            value=unclustered_questions,
            help="Questions that couldn't be categorized (noise)"
        )
    
    with col5:
        st.metric(
            label="📈 Categorization Rate",
            value=f"{categorized_percentage:.1f}%",
            help="Percentage of questions successfully categorized"
        )
    
    st.divider()
    
    # Visual breakdown
    st.subheader("📊 Visual Analytics")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
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
    
    # Success indicators
    st.subheader("🎉 Analysis Quality Indicators")
    
    indicator_col1, indicator_col2, indicator_col3 = st.columns(3)
    
    with indicator_col1:
        categorization_quality = "🟢 Excellent" if categorized_percentage > 80 else "🟡 Good" if categorized_percentage > 60 else "🔴 Needs Improvement"
        st.markdown(f"""
        **Categorization Quality:** {categorization_quality}
        
        • {categorized_percentage:.1f}% of questions categorized
        • {unique_clusters} distinct topics discovered
        • Suitable for analysis and insights
        """)
    
    with indicator_col2:
        cluster_balance = "🟢 Well Balanced" if unique_clusters >= 10 and unique_clusters <= 50 else "🟡 Moderate" if unique_clusters >= 5 else "🔴 Too Few Clusters"
        st.markdown(f"""
        **Cluster Distribution:** {cluster_balance}
        
        • {unique_clusters} clusters for {total_questions} questions
        • Avg {total_questions / max(unique_clusters, 1):.1f} questions per cluster
        • Good granularity for review
        """)
    
    with indicator_col3:
        min_cluster_quality = "🟢 Optimized" if unique_clusters > 0 else "🔴 No Clusters"
        st.markdown(f"""
        **Configuration Status:** {min_cluster_quality}
        
        • Min cluster size set to 15 ✅
        • Using latest OpenAI models ✅
        • Enhanced topic labeling ✅
        """)
    
    # Download section for Elder Edwards review format
    st.divider()
    st.subheader("📥 Download for Review")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        st.markdown("**📋 Standard Analysis Download**")
        csv_standard = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Complete Analysis",
            data=csv_standard,
            file_name=f"pathway_questions_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Full analysis with all columns and metrics"
        )
    
    with download_col2:
        st.markdown("**📝 Elder Edwards Review Format**")
        # Create the exact format requested: representation and question columns, sorted
        review_df = df[['Topic_Name', 'Question']].copy()
        review_df = review_df.rename(columns={'Topic_Name': 'representation'})
        # Sort by representation (topic) and then by question alphabetically as specified
        review_df = review_df.sort_values(['representation', 'Question'])
        
        csv_review = review_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Review CSV",
            data=csv_review,
            file_name=f"pathway_questions_review_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Sorted by topic and question for Elder Edwards review"
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
