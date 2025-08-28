import base64
import json
import logging
import numpy as np
import os
import pandas as pd
import requests
import streamlit as st
import time

from databricks.sdk import WorkspaceClient
from io import BytesIO
from PIL import Image
from mlflow.deployments import get_deploy_client


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For getting local images rather than serving endpoint (no custom input)
local_mode = False

# For generating image from Replicate (False = Returns vector search image as placeholder)
replicate_toggle = True

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

w = WorkspaceClient()

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(prompt, params=None):
    url = f"https://{os.environ.get('DATABRICKS_HOST')}/serving-endpoints/{os.environ.get('SERVING_ENDPOINT')}/invocations"
    headers = {
        "Authorization": f'Bearer {os.environ.get("SP_TOKEN")}', 
        "Content-Type": "application/json"
    }

    dataset = pd.DataFrame({"model_input": [prompt]})

    payload = {
        "dataframe_split": {
            "columns": list(dataset.columns),
            "data": dataset.values.tolist()
        }
    }
    if params:
        payload["params"] = params
    
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


# --- Session state flags ---
if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False
if "processing" not in st.session_state:
    st.session_state.processing = False

def on_generate_click():
    if not st.session_state.processing:
        st.session_state.generate_clicked = True
        st.session_state.processing = True

# --- Page Configuration ---
st.set_page_config(page_title="Pet Ad Image Gen", page_icon="🐾", layout="wide")
# Use tabs for cleaner navigation
# page_tabs = st.tabs(["Image Gen", "Chat Mode"])
page_tabs = st.tabs(["Image Gen"])

# Custom CSS for cleaner, modern tabs (less red, more neutral)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        background: #f5f5f7;
        color: #333 !important;
        font-weight: 600;
        font-size: 1.08em;
        border-radius: 8px 8px 0 0;
        box-shadow: 0 1px 6px #ddd;
        padding: 0.45em 1.5em;
        border: 1px solid #e0e0e0;
        transition: background 0.2s, color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #fff;
        color: #ff6f61 !important;
        box-shadow: 0 2px 10px #eee;
        border-bottom: 2px solid #ff6f61;
    }
</style>
""", unsafe_allow_html=True)

with page_tabs[0]:
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center;'>
            <h2 style='color: #ff6f61;'>🐾 Pet Ad Image Gen</h2>
            <p style='font-size: 1.1em;'>Create stunning pet ad creatives for your audience segments.<br><br>
            <b>How to use:</b><br>
            1. Select an audience segment<br>
            2. Click <span style='color:#ff6f61'><b>Generate Image</b></span><br>
            3. View the AI-generated ad and source image side by side<br>
            <!-- <span style='color:#ff6f61; font-size:1em; display:block; margin-top:1.5em; margin-bottom:1em;'><b>Switch to Chat Mode to interact with an AI Agent.</b></span> -->
            </p>
            <hr>
            <small style='color: #888;'>Powered by Databricks</small>
        </div>
        """, unsafe_allow_html=True)

    # --- Main Title ---
    st.markdown("""
    <div style='text-align: center; margin-bottom: 0.5em;'>
        <h1 style='color: #ff6f61; font-size: 2.7em; margin-bottom: 0;'>Pet Ad Image Gen App</h1>
        <span style='font-size: 2em;'>🐕&nbsp;🐈&nbsp;🐇&nbsp;🦜</span>
    </div>
    """, unsafe_allow_html=True)

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


    # --- Segment Selection ---
    segment_options = list(segments.keys())
    if not local_mode: # Only when not in 'local mode'
        segment_options.append("No Segment (custom prompt)")
    segment_name = st.selectbox("Choose an audience segment:", segment_options, format_func=lambda x: f"{x}", help="Select the target audience for your pet ad.")

    if segment_name == "No Segment (custom prompt)":
        st.markdown("### Custom Prompt")
        custom_prompt = st.text_input("Enter your custom prompt (required):")
        if not custom_prompt.strip():
            st.warning("Please enter a prompt to continue.")
        vector_query = None
        persona_summary = None

    else:
        segment = segments[segment_name]
        vector_query = segment["vector_query"]
        persona_summary = segment["persona_summary"]

        # --- Segment Info Card ---
        with st.container():
            st.markdown(f"""
            <div style='background: #fff7f0; border-radius: 14px; padding: 0.7em 1em; margin-bottom: 0.5em; box-shadow: 0 2px 12px #ff6f6133;'>
                <h3 style='color:#ff6f61; margin-bottom:0.15em; font-size:1.15em;'>🎯 Segment Information</h3>
                <p style='font-size:0.95em; margin-bottom:0.2em;'><b>Persona Summary:</b> {persona_summary}</p>
                <p style='font-size:0.9em; margin-bottom:0;'><b>Query:</b> <i>{vector_query}</i></p>
            </div>
            """, unsafe_allow_html=True)

    # --- Generate Button ---
    st.markdown("""
    <div style='text-align:center; margin-bottom:1em;'>
        <style>
        .stButton > button {
            font-size: 1.35em;
            font-weight: 700;
            padding: 0.7em 2.2em;
            border-radius: 12px;
            background: linear-gradient(90deg, #ff6f61 60%, #ffb88c 100%);
            color: white !important;
            border: none;
            box-shadow: 0 2px 12px #ff6f6133;
            transition: transform 0.1s, box-shadow 0.1s, background 0.1s;
        }
        .stButton > button:active {
            color: white !important;
            background: linear-gradient(90deg, #ff6f61 60%, #ffb88c 100%);
        }
        .stButton > button:hover {
            transform: scale(1.07);
            box-shadow: 0 4px 18px #ff6f6166;
            background: linear-gradient(90deg, #ffb88c 60%, #ff6f61 100%);
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.button("✨ Generate Image", on_click=on_generate_click, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Main execution block
    if st.session_state.generate_clicked and st.session_state.processing:
        # Immediately reset flag to prevent reruns
        st.session_state.generate_clicked = False

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
                    time.sleep(6) # Simulate longer delay for realism
                    original_image = Image.open(local_pet_images[segment_name])
                    generated_image = Image.open(local_ad_images[segment_name])
                else:
                    # now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    # st.write("making agent call:", now)

                    response = score_model(prompt, params={"replicate_toggle": replicate_toggle})

                    # original_image = Image.open(BytesIO(base64.b64decode(response['predictions'][1])))
                    original_image_path = response['predictions'][1]
                    download_response = w.files.download(original_image_path)
                    file_data = download_response.contents.read()
                    original_image = Image.open(BytesIO(file_data))

                    generated_image = Image.open(BytesIO(base64.b64decode(response['predictions'][2])))

                st.markdown("""
                <div style='background: #fff7f0; border-radius: 14px; padding: 0.7em 1em; margin-bottom: 0.5em; box-shadow: 0 2px 12px #ff6f6133;'>
                    <h3 style='color:#ff6f61; margin-bottom:0.15em; font-size:1.15em;'>🖼️ Generated Ad & Source Image</h3>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])
                max_img_height = 1024
                def show_image(img, caption):
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    st.markdown(f"""
                    <div style='text-align:center;'>
                        <img src='data:image/png;base64,{img_b64}' style='max-height:{max_img_height}px; width:auto; border-radius:10px; box-shadow:0 1px 8px #ff6f6133;'/>
                        <div style='font-size:0.95em; color:#555; margin-top:0.3em;'>{caption}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col1:
                    show_image(generated_image, "✨ AI-generated ad creative")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    show_image(original_image, "🔍 Retrieved from vector search")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.session_state.processing = False

            except Exception as e:
                st.error(f"An error occurred: {e}")

# with page_tabs[1]:
#     st.markdown("""
#     <div style='text-align:center; margin-top:3em;'>
#         <h2 style='color:#ff6f61;'>💬 Chat Mode</h2>
#         <p style='font-size:1.1em; color:#555;'>This is a placeholder for the Chat Mode page.</p>
#     </div>
#     """, unsafe_allow_html=True)
