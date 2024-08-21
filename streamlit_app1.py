import streamlit as st
import pandas as pd
import numpy as np

# Title of the dashboard
st.title('Simple Streamlit Dashboard')

# Sidebar for user input
st.sidebar.header('User Input')
user_input = st.sidebar.text_input('Enter a message', 'Hello, Streamlit!')

# Display user input
st.write('User Input:', user_input)

# Generate some data
data = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100)
})

# Display the data
st.write('Data:')
st.dataframe(data)

# Plot the data
st.line_chart(data)

# Add a map
st.map(pd.DataFrame({
    'lat': np.random.randn(100) / 50 + 37.76,
    'lon': np.random.randn(100) / 50 - 122.4
}))

# Add a button
if st.button('Click me'):
    st.write('Button clicked!')

# Add a slider
slider_value = st.slider('Select a value', 0, 100, 50)
st.write('Slider value:', slider_value)
