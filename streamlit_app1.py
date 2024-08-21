import streamlit as st
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

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
    
    # Display the dataframe
    st.write('Sentiment Analysis Results:')
    st.dataframe(sentiment_df)
    
    # Plot the sentiment scores
    st.write('Sentiment Scores Visualization:')
    fig, ax = plt.subplots()
    sentiment_df[['Positive', 'Neutral', 'Negative']].mean().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    
    # Save the results to an Excel file
    output_file_path = os.path.join(os.path.dirname(uploaded_file.name), 'sentiment_analysis_results.xlsx')
    sentiment_df.to_excel(output_file_path, index=False)
    st.write(f'Sentiment analysis results have been saved to {output_file_path}')
else:
    st.write('Please upload an Excel file to see the data and visualization.')
