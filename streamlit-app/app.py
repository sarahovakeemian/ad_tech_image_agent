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
# assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

# --- NEW: Add a session state flag for button click SH added---
if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False

if "processing" not in st.session_state:
    st.session_state.processing = False

def on_generate_click():
    if not st.session_state.processing:
        st.session_state.generate_clicked = True
        st.session_state.processing = True

st.title("🐶🐱 Pet Ad Image Gen App")

st.image("images/arch_diagram_V2.png", use_container_width=True, caption="Databricks Architecture")

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
    },
    "Luxury Lifestyle Seekers": {
        "vector_query": "White poodle",
        "persona_summary": "Values sophistication and pampering, treating their pet as a stylish companion and status symbol."
    },
    "Security-Focused Guardians": {
        "vector_query": "Tough rottweiler",
        "persona_summary": "Seeks a protective companion that embodies strength while maintaining family loyalty."
    },
    "Vocal Companion Enthusiasts": {
        "vector_query": "Quaker Parrot",
        "persona_summary": "Loves interactive pets that provide lively companionship and engaging dialogue."
    },
    "Traditional Loyalty Advocates": {
        "vector_query": "Old Bulldog",
        "persona_summary": "Cherishes steadfast companionship and the quiet dignity of a time-tested breed."
    },
    "Style-Conscious Pet Parents": {
        "vector_query": "Siamese kittens",
        "persona_summary": "Adores elegant felines that complement their modern aesthetic while providing playful energy."
    },
    "Mystical Nature Admirers": {
        "vector_query": "Black forest cat",
        "persona_summary": "Drawn to mysterious feline companions that evoke woodland magic and quiet independence."
    }
}

# Helpers to map segment to local images
local_pet_images = {
    "Young Professionals": "images/grey_cat.jpg",
    "Outdoor Enthusiasts": "images/golden_retriever.jpg",
    "Young Families": "images/guinea_pig.jpg",
    "Eco-Conscious Millenials": "images/rabbit.jpg",
    "Passionate Hobbyists": "images/parrot.jpg",
    "Luxury Lifestyle Seekers":"images/poodle.png",
    "Security-Focused Guardians":"images/rottweiler.png",
    "Vocal Companion Enthusiasts":"images/quaker_parrot.png",
    "Traditional Loyalty Advocates":"images/bulldog.png",
    "Style-Conscious Pet Parents":"images/siamese_kittens.png",
    "Mystical Nature Admirers":"images/black_forest_cat.png"
}
local_ad_images = {
    "Young Professionals": "images/grey_cat_ad.jpg",
    "Outdoor Enthusiasts": "images/golden_retriever_ad.jpg",
    "Young Families": "images/guinea_pig_ad.jpg",
    "Eco-Conscious Millenials": "images/rabbit_ad.jpg",
    "Passionate Hobbyists": "images/parrot_ad.jpg",
    "Luxury Lifestyle Seekers":"images/poodle_ad.png",
    "Security-Focused Guardians":"images/rottweiler_ad.png",
    "Vocal Companion Enthusiasts":"images/quaker_parrot_ad.png",
    "Traditional Loyalty Advocates":"images/bulldog_ad.png",
    "Style-Conscious Pet Parents":"images/siamese_kittens_ad.png",
    "Mystical Nature Admirers":"images/black_forest_cat_ad.png"
}

segment_options = list(segments.keys())
segment_name = st.selectbox("Choose an audience segment:", segment_options)

segment = segments[segment_name]
custom_prompt = None  # ignore user input in this mode
vector_query = segment["vector_query"]
persona_summary = segment["persona_summary"]

st.markdown("### Segment Information")
st.write(f"**Persona Summary:** {persona_summary}")
st.markdown(f"**Query:** *{vector_query}*")

st.button("Generate Image", on_click=on_generate_click) #SH added---

# Main execution block - now OUTSIDE the button check
if st.session_state.generate_clicked and st.session_state.processing:
    # Immediately reset flag to prevent reruns
    st.session_state.generate_clicked = False

    prompt = vector_query
    use_endpoint = not local_mode  # Follow local_mode for predefined segments

    with st.spinner("Generating image..."):
        now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.write("spinner:", now)
        try:
            time.sleep(2) # Simulate delay
            original_image = Image.open(local_pet_images[segment_name])
            generated_image = Image.open(local_ad_images[segment_name])

            st.markdown("### Generated Ad and Source Image")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.image(generated_image, use_container_width=True, caption="✨ AI-generated ad creative")

            with col2:
                st.image(original_image, use_container_width=True, caption="🐾 Retrieved from vector search")

            st.session_state.processing = False

        except Exception as e:
            st.error(f"An error occurred: {e}")
            