import streamlit as st
import pandas as pd
import os
from PIL import Image

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
# print current dir
print(os.getcwd())
# Specify the directory containing the CSV files
csv_directory = "sae_analysis/out/new/"  # Update with your CSV folder path

# Get a list of feature indices (based on available CSV files)
csv_files = {int(f.split('_')[0][1:]): os.path.join(csv_directory, f) for f in os.listdir(csv_directory) if f.endswith('_stats.csv')}
feature_indices = sorted(csv_files.keys())[1:]


# Searchable dropdown for feature index selection
selected_idx = st.sidebar.selectbox("Select Feature Index", feature_indices)

# Load the data for the selected feature index
if selected_idx:
    file_path = csv_files[selected_idx]
    data = load_data(file_path)

    # Display the numerical columns
    st.header(f"Feature Index {selected_idx} - Data Table")

    st.write("Semantic Description: In progress...")

    numerical_columns = data.select_dtypes(include=['number'])
    st.dataframe(numerical_columns)

    # Display the image pairs
    if 'img_path_in' in data.columns and 'img_path_out' in data.columns and selected_idx not in [2, 922]:
        st.header("Image Pairs (Left: Input Actions, Right: Output Actions)")
        image_paths1 = data['img_path_in'].tolist()
        image_paths2 = data['img_path_out'].tolist()
        display_image_pairs(image_paths1, image_paths2)
    else:
        st.warning("Image columns not found in this dataset.")
