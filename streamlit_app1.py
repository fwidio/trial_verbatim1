import streamlit as st
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Initialize the sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()

# Title of the app
st.title('Sentiment Analysis and Visualization')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    
    # Perform sentiment analysis on each comment
    results = []
    for comment in df['comment']:
        if isinstance(comment, str):
            sentiment_scores = analyzer.polarity_scores(comment.strip())
            results.append({
                'Comment': comment.strip(),
                'Compound': sentiment_scores['compound'],
                'Positive': sentiment_scores['pos'],
                'Neutral': sentiment_scores['neu'],
                'Negative': sentiment_scores['neg']
            })
        else:
            results.append({
                'Comment': comment,
                'Compound': None,
                'Positive': None,
                'Neutral': None,
                'Negative': None
            })
    
    # Convert the results to a DataFrame
    sentiment_df = pd.DataFrame(results)
    
    # Filter options
    filter_option = st.selectbox('Filter by Sentiment', ['All', 'Positive', 'Neutral', 'Negative'])
    
    if filter_option == 'Positive':
        filtered_df = sentiment_df[sentiment_df['Positive'] > sentiment_df[['Positive', 'Neutral', 'Negative']].max(axis=1)]
    elif filter_option == 'Neutral':
        filtered_df = sentiment_df[sentiment_df['Neutral'] > sentiment_df[['Positive', 'Neutral', 'Negative']].max(axis=1)]
    elif filter_option == 'Negative':
        filtered_df = sentiment_df[sentiment_df['Negative'] > sentiment_df[['Positive', 'Neutral', 'Negative']].max(axis=1)]
    else:
        filtered_df = sentiment_df
    
    # Display the filtered dataframe
    st.write('Sentiment Analysis Results:')
    st.dataframe(filtered_df)
    
    # Plot the sentiment scores using Plotly
    st.write('Sentiment Scores Visualization:')
    
    # Bar chart for average sentiment scores
    avg_scores = sentiment_df[['Positive', 'Neutral', 'Negative']].mean().reset_index()
    avg_scores.columns = ['Sentiment', 'Score']
    fig = px.bar(avg_scores, x='Sentiment', y='Score', title='Average Sentiment Scores')
    
    # Add click event to filter table
    fig.update_traces(marker=dict(color=['blue', 'orange', 'red']))
    fig.update_layout(clickmode='event+select')
    
    st.plotly_chart(fig)
    
    # Scatter plot for compound scores
    fig = px.scatter(sentiment_df, x='Comment', y='Compound', title='Compound Sentiment Scores')
    st.plotly_chart(fig)
    
    # Save the results to an Excel file
    output_file_path = os.path.join(os.path.dirname(uploaded_file.name), 'sentiment_analysis_results.xlsx')
    sentiment_df.to_excel(output_file_path, index=False)
    st.write(f'Sentiment analysis results have been saved to {output_file_path}')
else:
    st.write('Please upload an Excel file to see the data and visualization.')
