import base64
import logging
import os
import streamlit as st
import time

from io import BytesIO
from PIL import Image
from mlflow.deployments import get_deploy_client


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For testing
use_local_image = False

if not use_local_image:
    # Initialize the Databricks Client
    client = get_deploy_client("databricks")

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

# def get_user_info():
#     headers = st.context.headers
#     return dict(
#         user_name=headers.get("X-Forwarded-Preferred-Username"),
#         user_email=headers.get("X-Forwarded-Email"),
#         user_id=headers.get("X-Forwarded-User"),
#     )

# user_info = get_user_info()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

            
st.title("🐱 Cat Ad Image Gen App")

# Segment definitions
segments = {
    "Suburban Home Owners": {
        "vector_query": "black cat prowling in a garden",
        "persona_summary": "Design-conscious suburban owners with an active outdoor cat."
    },
    "Adventure Seekers": {
        "vector_query": "ginger cat playing in the snow",
        "persona_summary": "Outdoor-loving couples who take their cat on seasonal trips."
    },
    "City Creators": {
        "vector_query": "tuxedo cat in the city",
        "persona_summary": "Trendy, urban, social media savvy cat owners."
    },
    "Homebodies": {
        "vector_query": "sleeping brown cat on chair",
        "persona_summary": "Indoorsy types who love to cozy up at home with a blanket and their cat."
    },
    "Luxury lovers": {
        "vector_query": "elegant white cat",
        "persona_summary": "Artsy luxury lovers with a stylish cat to show off."
    }
}

# Helper to map segment to local cat image
local_cat_images = {
    "Suburban Home Owners": "test_images/cat_01.png",
    "Adventure Seekers": "test_images/cat_02.png",
    "City Creators": "test_images/cat_03.png",
    "Homebodies": "test_images/cat_04.png",
    "Luxury lovers": "test_images/cat_05.png",
}

segment_options = list(segments.keys()) + ["No Segment (custom prompt)"]
segment_name = st.selectbox("Choose an audience segment:", segment_options)

if segment_name == "No Segment (custom prompt)":
    st.markdown("### Custom Prompt")
    custom_prompt = st.text_input("Enter your custom prompt (required):")
    if not custom_prompt.strip():
        st.warning("Please enter a prompt to continue.")
    vector_query = None
    persona_summary = None
else:
    segment = segments[segment_name]
    custom_prompt = None  # ignore user input in this mode
    vector_query = segment["vector_query"]
    persona_summary = segment["persona_summary"]

    st.markdown("### Segment Information")
    st.write(f"**Persona Summary:** {persona_summary}")
    st.markdown(f"**Query:** *{vector_query}*")

if st.button("Generate Image"):
    if segment_name == "No Segment (custom prompt)":
        prompt = custom_prompt.strip()
        if not prompt:
            st.error("Custom prompt is required when not using a segment.")
            st.stop()
    else:
        prompt = vector_query
    with st.spinner("Generating image..."):
        try:
            if use_local_image:
                time.sleep(2)  # Simulate delay
                original_image = Image.open(local_cat_images[segment_name])
                generated_image = Image.open("test_images/image_gen_example.png")
            else:
                response = client.predict(
                    endpoint=os.getenv("SERVING_ENDPOINT"),
                    inputs={
                        "dataframe_split": {
                            "columns": ["model_input"],
                            "data": [[prompt]]
                        }
                    }
                )

                original_image = Image.open(BytesIO(base64.b64decode(response['predictions'][1])))
                generated_image = Image.open(BytesIO(base64.b64decode(response['predictions'][2])))

            st.markdown("### Generated Ad and Source Image")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.image(generated_image, use_container_width=True, caption="✨ AI-generated ad creative")

            with col2:
                st.image(original_image, use_container_width=True, caption="🐈 Retrieved from vector search")

        except Exception as e:
            st.error(f"An error occurred: {e}")
