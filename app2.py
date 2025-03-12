import streamlit as st
import requests
from PIL import Image
import io

# Replace with your actual Azure Custom Vision credentials and details.
PREDICTION_KEY = "1fd3f87d5f424bb8bdcef6e5af3e7d9e"
ENDPOINT = "https://drowsiness-prediction.cognitiveservices.azure.com"  # e.g., https://<your-resource-name>.cognitiveservices.azure.com
PROJECT_ID = "cd25d6ad-aaf9-4be9-ada3-2529205b67f5"
ITERATION_NAME = "Iteration2"  # The published model iteration name

def get_prediction(image_bytes, prediction_key, endpoint, project_id, iteration_name):
    headers = {
        "Prediction-Key": prediction_key,
        "Content-Type": "application/octet-stream"
    }
    url = f"{endpoint}/customvision/v3.0/Prediction/{project_id}/classify/iterations/{iteration_name}/image"
    response = requests.post(url, headers=headers, data=image_bytes)
    response.raise_for_status()
    return response.json()

st.title("Azure Custom Vision Prediction Bot")

# Image uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Open and display the uploaded image.
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Predicting...")
    
    # Convert the image to bytes.
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format=image.format)
    img_bytes = img_bytes_io.getvalue()
    
    # Get predictions from the Custom Vision API.
    predictions = get_prediction(img_bytes, PREDICTION_KEY, ENDPOINT, PROJECT_ID, ITERATION_NAME)
    
    # Process and display the predictions.
    st.write("### Predictions:")
    for prediction in predictions.get("predictions", []):
        tag = prediction.get("tagName", "N/A")
        probability = prediction.get("probability", 0) * 100  # convert to percentage
        st.write(f"{tag}: {probability:.2f}%")
        
    # Optionally, decide on a "yes" or "no" based on the highest probability:
    if predictions.get("predictions"):
        best_prediction = max(predictions["predictions"], key=lambda p: p["probability"])
        st.write(f"\n*Final Decision:* {best_prediction['tagName']} with {best_prediction['probability']*100:.2f}% accuracy.")