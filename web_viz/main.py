import streamlit as st
import pandas as pd
import os
from PIL import Image
import json

# Function to load data from a selected CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to display images in pairs
def display_image_pairs(image_paths1, image_paths2):
    for img1, img2 in zip(image_paths1, image_paths2):
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1)
        with col2:
            st.image(img2)

# Set up the Streamlit interface
st.title("Diffusion Policy SAE Feature Viz")
st.write("Sparse Autoencoders are a way to look inside the model and find interpretable units of computation that the model uses. Each unit/feature is supposed to be interpretable and monosemantic (representing a single thing).")
st.write("The tables are numerical analysis of the environment and actions (angle, distance, etc) while the image pairs show the environment states where it activates the most.")
st.sidebar.title("Options")

# Print current directory (for debugging)
print(os.getcwd())

# Specify the directory containing the CSV files
csv_directory = "sae_analysis/out/new/"  # Update with your CSV folder path

# Get a list of feature indices (based on available CSV files)
csv_files = {int(f.split('_')[0][1:]): os.path.join(csv_directory, f) for f in os.listdir(csv_directory) if f.endswith('_stats.csv')}
feature_indices = sorted(csv_files.keys())[1:]

# Read the descriptions JSON
with open('sae_analysis/out/descriptions/new.json') as f:
    descriptions = json.load(f)

# Define featured feature indices
featured_indices = [7, 206, 922, 1198, 2043, 2037, 2029, 2025]  # Replace with your desired indices
st.sidebar.write("### Featured Feature Indices")

# Handle clicks for featured indices
clicked_feature = None
for idx in featured_indices:
    if st.sidebar.button(f"Feature {idx}"):
        clicked_feature = idx

# Searchable dropdown for general feature index selection
st.sidebar.write("### All Feature Indices")
selected_idx = st.sidebar.selectbox("Select Feature Index", feature_indices)
# Determine which feature index is selected
if clicked_feature:
    selected_idx = clicked_feature # Prioritize featured index selection

# Load the data for the selected feature index
if selected_idx:
    file_path = csv_files[selected_idx]
    data = load_data(file_path)

    # Display the numerical columns
    st.header(f"Feature Index {selected_idx} - Data Table")

    if str(selected_idx) in descriptions:
        st.write(f"Semantic Description: {descriptions[str(selected_idx)]}")
    else:
        st.write("Semantic Description: In progress...")

    numerical_columns = data.select_dtypes(include=['number'])
    st.dataframe(numerical_columns)

    st.header("Steering")
    # if st.button("Steer this feature!"):
    #     pass
    st.write("In progress, I dont know how to run the models in streamlit yet.")

    # Display the image pairs
    if 'img_path_in' in data.columns and 'img_path_out' in data.columns and selected_idx not in [2]:
        st.header("Image Pairs (Left: Input Actions, Right: Output Actions)")
        image_paths1 = data['img_path_in'].tolist()
        image_paths2 = data['img_path_out'].tolist()
        display_image_pairs(image_paths1, image_paths2)
    else:
        st.warning("Image columns not found in this dataset.")

