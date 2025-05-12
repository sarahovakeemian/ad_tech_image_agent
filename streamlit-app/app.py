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

# For getting local images rather than serving endpoint
local_mode = True

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

            
st.title("🐶🐱 Pet Ad Image Gen App")

# Segment definitions
segments = {
    "Young Professionals": {
        "vector_query": "Grey cat on a rug",
        "persona_summary": "A driven urban worker who values independence, modern living, and the quiet companionship of their furry friend."
    },
    "Outdoor Enthusiasts": {
        "vector_query": "Golden retriever in the park",
        "persona_summary": "An active individual who thrives on adventure, fresh air, and sharing it all with their loyal friend by their side."
    },
    "Young Families": {
        "vector_query": "Guinea pig in it’s cage",
        "persona_summary": "Creates a nurturing home where their children are learning care and responsibility through their very first pet."
    },
    "Eco-Conscious Millenials": {
        "vector_query": "Bunny rabbits in the grass",
        "persona_summary": "Lives sustainably and intentionally, choosing a pet as a low-impact companion that aligns with their eco-friendly values."
    },
    "Passionate Hobbyists": {
        "vector_query": "Bright green parrots",
        "persona_summary": "A retired hobbyist who fills their days with passion projects and lively conversations with their talkative pet."
    }
}

# Helpers to map segment to local images
local_pet_images = {
    "Young Professionals": "images/cat.jpg",
    "Outdoor Enthusiasts": "images/dog.jpg",
    "Young Families": "images/guinea_pig.jpg",
    "Eco-Conscious Millenials": "images/rabbit.jpg",
    "Passionate Hobbyists": "images/parrot.jpg",
}
local_ad_images = {
    "Young Professionals": "images/cat_ad.jpg",
    "Outdoor Enthusiasts": "images/dog_ad.jpg",
    "Young Families": "images/guinea_pig_ad.jpg",
    "Eco-Conscious Millenials": "images/rabbit_ad.jpg",
    "Passionate Hobbyists": "images/parrot_ad.jpg",
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
        use_endpoint = True  # Force endpoint for custom prompt
    else:
        prompt = vector_query
        use_endpoint = not local_mode  # Follow local_mode for predefined segments

    with st.spinner("Generating image..."):
        try:
            if not use_endpoint:
                time.sleep(5) # Simulate delay
                original_image = Image.open(local_pet_images[segment_name])
                generated_image = Image.open(local_ad_images[segment_name])
            else:
                # Initialize the Databricks Client
                client = get_deploy_client("databricks")
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
