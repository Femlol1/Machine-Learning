import sys

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import datetime

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
import torch
import torch.nn.functional as F
from config import DB_CONFIG
from model import MNISTNet
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# --- Load model ---
device = torch.device("cpu")
model = MNISTNet().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# --- Helpers ---
def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def log_prediction(predicted: int, true_label: int):
    ts = datetime.datetime.utcnow()
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (timestamp, predicted_digit, true_label) VALUES (%s, %s, %s)",
            (ts, predicted, true_label)
        )
        conn.commit()
    except Exception as e:
        st.error(f"DB logging failed: {e}")
    finally:
        cur.close()
        conn.close()

# --- App UI ---
st.title("MNIST Digit Recognizer")
st.write("Draw a digit below:")

canvas_result = st_canvas(
    fill_color="#000",
    stroke_width=10,
    stroke_color="#fff",
    background_color="#000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    pil_img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA").convert("RGB")
    st.image(pil_img, caption="Your Drawing", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        tensor = preprocess_image(pil_img)
        out = model(tensor)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        # store in session state
        st.session_state.prediction = int(pred)
        st.session_state.confidence = float(conf)

    # If we have a prediction, show it (and the feedback UI)
    if "prediction" in st.session_state:
        st.write(f"**Prediction:** {st.session_state.prediction}")
        st.write(f"**Confidence:** {st.session_state.confidence*100:.2f}%")

        # Feedback form
        true = st.text_input("Enter the true digit:", key="true_label")
        if st.button("Submit Feedback"):
            try:
                lbl = int(st.session_state.true_label)
                log_prediction(st.session_state.prediction, lbl)
                st.success("Feedback logged! 🎉")
                # Optionally clear state so form goes away:
                del st.session_state.prediction
                del st.session_state.confidence
                del st.session_state.true_label
            except ValueError:
                st.error("Please enter a valid integer 0–9.")
                
def fetch_logs(limit: int = 100):
    """Retrieve the most recent `limit` prediction records."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql_query(
            """
            SELECT id, timestamp, predicted_digit, true_label
            FROM predictions
            ORDER BY id DESC
            LIMIT %s
            """,
            conn,
            params=(limit,)
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")
        return pd.DataFrame()  # empty

# --- Logs section ---

st.header("Prediction Logs")

# Allow the user to choose how many rows to view
num_rows = st.number_input(
    "How many recent entries to show?",
    min_value=10, max_value=1000, value=100, step=10
)

if st.button("Refresh Logs"):
    st.session_state.logs = fetch_logs(num_rows)

# Initialize on first run
if "logs" not in st.session_state:
    st.session_state.logs = fetch_logs(num_rows)

# Display the DataFrame
st.dataframe(
    st.session_state.logs,
    use_container_width=True
)
