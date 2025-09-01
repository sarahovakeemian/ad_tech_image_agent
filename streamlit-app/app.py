import base64
import json
import logging
import mlflow
import numpy as np
import os
import pandas as pd
import random
import streamlit as st
import time

from databricks.sdk import WorkspaceClient
from io import BytesIO
from PIL import Image
from mlflow.deployments import get_deploy_client
from segments import segments, local_pet_images, local_ad_images


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For getting local images rather than serving endpoint (no custom input)
local_mode = False

# For generating image from Replicate (False = Returns vector search image as placeholder)
replicate_toggle = True

# Ensure environment variable is set correctly
assert os.getenv('AGENT_ENDPOINT'), "AGENT_ENDPOINT must be set in app.yaml."

AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "")
client = get_deploy_client("databricks")


# --- Helpers to get output text and images from ResponseAgent --- #
def call_agent(messages):
    """Call the agent (Responses schema only) and return the raw response dict."""
    payload_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("content") and str(m["content"]).strip()
    ]
    resp = client.predict(endpoint=AGENT_ENDPOINT, inputs={"input": payload_msgs})
    if not isinstance(resp, dict):
        raise RuntimeError("Agent returned a non-dict response.")
    return resp

def extract_output_texts(resp: dict) -> list[str]:
    """Collect assistant text from Responses-style outputs (message/output_text)."""
    items = resp.get("output", []) or []
    texts: list[str] = []
    for it in items:
        t = it.get("type")
        if t == "output_text":
            txt = it.get("text")
            if txt:
                texts.append(txt)
        elif t == "message":
            for part in (it.get("content") or []):
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    txt = part.get("text")
                    if txt:
                        texts.append(txt)
    return texts

@st.cache_resource
def get_ws():
    return WorkspaceClient()

@st.cache_data(show_spinner=False)
def load_uc_image_bytes(path: str, max_h: int | None = None) -> bytes:
    raw = get_ws().files.download(path).contents.read()
    img = Image.open(BytesIO(raw)).convert("RGBA")
    if max_h and img.height > max_h:
        ratio = max_h / float(img.height)
        img = img.resize((int(img.width * ratio), max_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

def extract_pointer(resp: dict):
    co = resp.get("custom_outputs") or {}
    ptr = co.get("last_image")
    if isinstance(ptr, dict) and ptr.get("type") in ("final", "seed"):
        return {
            "type": ptr["type"],
            "uc_path": ptr.get("uc_path"),
            "seed_path": ptr.get("seed_path"),
        }
    # Fallback: parse last tool output if needed
    items = resp.get("output", []) or []
    for it in reversed(items):
        if it.get("type") == "function_call_output":
            try:
                payload = json.loads(it.get("output") or "{}")
            except Exception:
                payload = {}
            if payload.get("generated"):
                return {"type": "final", "uc_path": (payload.get("result") or {}).get("uc_path"), "seed_path": None}
            return {"type": "seed", "uc_path": None, "seed_path": payload.get("retrieved_uc_path")}
    return None

def pointer_to_image(ptr: dict | None):
    if not ptr:
        return None, None
    try:
        if ptr.get("type") == "final" and ptr.get("uc_path"):
            return load_uc_image_bytes(ptr["uc_path"]), "✨ AI-generated ad creative"
        if ptr.get("type") == "seed" and ptr.get("seed_path"):
            sp = ptr.get("seed_path") or ptr.get("uc_path")
            if sp:
                return load_uc_image_bytes(sp, max_h=512), "🔍 Retrieved from vector search"
    except Exception as e:
        st.warning(f"Couldn't load image ({e}).")
    return None, None


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
page_tabs = st.tabs(["Image Gen", "Chat Mode"])
# page_tabs = st.tabs(["Image Gen"])

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
            <p style='font-size: 1.1em;'>Create stunning ad creatives for Bricks™ pet food, personalized to your audience segments.<br><br>
            <b>How to use:</b><br>
            1. Select an audience segment<br>
            2. Click <span style='color:#ff6f61'><b>Generate Image</b></span><br>
            3. View the AI-generated ad and source image side by side<br>
            <span style='color:#ff6f61; font-size:1em; display:block; margin-top:1.5em; margin-bottom:1em;'><b>Switch to <i>Chat Mode</i> to interact with an AI Agent.</b></span>
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
            use_endpoint = True  # Only custom prompt uses the agent
        else:
            prompt = vector_query
            use_endpoint = False # Pre-defined segments never query the endpoint

        with st.spinner("Generating image..."):
            try:
                if not use_endpoint:
                    # Local demo mode
                    time.sleep(6) # Simulate longer delay for realism
                    original_image = Image.open(local_pet_images[segment_name])
                    generated_image = Image.open(local_ad_images[segment_name])
                
                else:
                    # Direct-mode call to the chat agent (one shot; returns seed + optional final)
                    resp = client.predict(
                        endpoint=AGENT_ENDPOINT,
                        inputs={
                            "input": [],  # no chat — direct mode
                            "custom_inputs": {
                                "direct": {
                                    "prompt": prompt,
                                    "replicate_toggle": bool(replicate_toggle),
                                }
                            },
                        },
                    )

                    # Parse the last tool output to get UC paths
                    items = resp.get("output", []) or []
                    tool_items = [it for it in items if it.get("type") == "function_call_output"]
                    if not tool_items:
                        raise RuntimeError("Agent returned no tool output (function_call_output).")

                    tool_json_str = tool_items[-1].get("output") or "{}"
                    ptr = json.loads(tool_json_str)

                    seed_path = ptr.get("retrieved_uc_path")
                    final_uc  = (ptr.get("result") or {}).get("uc_path") if ptr.get("generated") else None

                    if not seed_path:
                        raise RuntimeError("Agent did not return a retrieved_uc_path for the seed image.")

                    # Fetch images from UC Volumes
                    seed_bytes = load_uc_image_bytes(seed_path)
                    original_image = Image.open(BytesIO(seed_bytes))

                    if final_uc:
                        final_bytes = load_uc_image_bytes(final_uc)
                        generated_image = Image.open(BytesIO(final_bytes))
                    else:
                        # If replicate off, reuse seed as “generated” placeholder
                        generated_image = original_image

                # --- Render result cards ---
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
                st.session_state.processing = False
                st.error(f"An error occurred: {e}")

with page_tabs[1]:
    st.markdown("""
    <style>
      .center-wrap { max-width: 820px; margin: 0 auto; text-align: center; }
      .chat-wrap   { max-width: 820px; margin: 1rem auto 0 auto; }
    </style>
    """, unsafe_allow_html=True)

    # Loading dots custom CSS
    st.markdown("""
    <style>
        :root{
            --avatar-size: 36px;
            --dot-size:   8px;
            --dot-gap:    6px;
            --nudge-y:   -8px;
        }

        .typing-bubble{
            display:flex;
            align-items:center;
            height: var(--avatar-size);
            line-height: 0;
        }
        .typing-dots{
            display:inline-flex;
            gap: var(--dot-gap);
            align-items:center;
            transform: translateY(var(--nudge-y));
        }
        .typing-dots span{
            width: var(--dot-size);
            height: var(--dot-size);
            border-radius:50%;
            background:#c9c9c9;
            display:inline-block;
            animation: typingBlink 1.4s infinite both;
        }
        .typing-dots span:nth-child(2){ animation-delay:.2s; }
        .typing-dots span:nth-child(3){ animation-delay:.4s; }

        @keyframes typingBlink { 0%,80%,100% { opacity:0; } 40% { opacity:1; } }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='center-wrap' style='margin-top:2rem; margin-bottom:2rem;'>
        <h2 style='color:#ff6f61;margin-bottom:0.25em;'>💬 Chat Mode</h2>
        <p style='font-size:1.05em; color:#555; margin:0;'>
            Describe your audience or describe a pet. I’ll fetch a seed image and, on confirmation, generate the final ad.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Config / endpoint name from env --- #
    if not AGENT_ENDPOINT:
        st.markdown("<div class='center-wrap'>", unsafe_allow_html=True)
        st.error("Set environment variable **AGENT_ENDPOINT** to your agent serving endpoint name (e.g., `pet-ad-agent`).")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ---- Avatars ---- #
    human_avatar = "👤"
    assistant_avatar = "🐾"

    # ---- Chat history ---- #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_area = st.container()

    # ---- Display chat history ---- #
    with chat_area:
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar=m.get("avatar", "")):
                if m.get("content"):
                    st.markdown(m["content"])
                if img_bytes := m.get("image_bytes"):
                    st.image(Image.open(BytesIO(img_bytes)), caption=m.get("image_caption", None))

    # ---- Accept user input ---- #
    prompt = st.chat_input("How can I help with your ad campaign?")
    if prompt:
        # Add the user message to history and render it now (history renderer above stays unchanged)
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": human_avatar})
        with chat_area:
            with st.chat_message("user", avatar=human_avatar):
                st.markdown(prompt)

            # Show loading dots
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.markdown(
                    "<div class='typing-bubble'><div class='typing-dots'><span></span><span></span><span></span></div></div>",
                    unsafe_allow_html=True
                )

        # Call the agent (no inline assistant bubble to avoid duplication)
        img_bytes, img_caption = None, None
        try:
            resp = call_agent(st.session_state.messages)
            texts = extract_output_texts(resp)
            assistant_response = "\n\n".join(texts) if texts else "(working on your request)"
            ptr = extract_pointer(resp)
            # (Optional) fetch image bytes now so they persist with the same assistant turn
            img_bytes, img_caption = pointer_to_image(ptr)
        except Exception as e:
            assistant_response = f"Sorry—there was an error talking to the agent: {e}"
            img_bytes, img_caption = None, None

        # Persist assistant message (text + optional image) to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "avatar": assistant_avatar,
            "image_bytes": img_bytes if img_bytes else None,
            "image_caption": img_caption if img_bytes else None,
        })

        # Rerun so the assistant turn renders once via the existing history loop
        st.rerun()
