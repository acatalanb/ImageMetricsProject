import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
from model import build_model
from metrics_manager import MetricsManager
from train import train_model_pipeline

# --- Config ---
st.set_page_config(page_title="Medical AI Manager", layout="wide", page_icon="🧬")

metrics_mgr = MetricsManager()
IMG_SIZE = 150
CACHE_DIR = 'cache'  # <--- NEW CONSTANT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create cache dir if it doesn't exist (to avoid errors on fresh run)
os.makedirs(CACHE_DIR, exist_ok=True)


# --- Helper Functions ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@st.cache_resource
def load_pytorch_model(model_name, path):
    try:
        model = build_model(model_name, pretrained=False)
        state_dict = torch.load(path, map_location=device)

        # Multi-GPU fix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        return None


def predict_single(model, image):
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    return prob, time.time() - start_time


def evaluate_test_set(model, test_dir, progress_bar):
    y_true, y_probs = [], []
    categories = ['NORMAL', 'PNEUMONIA']
    transform = get_transform()

    total_files = sum([len(files) for r, d, files in os.walk(test_dir)])
    processed = 0
    metrics_mgr.start_timer()

    for label_idx, category in enumerate(categories):
        folder_path = os.path.join(test_dir, category)
        if not os.path.exists(folder_path): continue
        for fname in os.listdir(folder_path):
            try:
                fpath = os.path.join(folder_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(img_tensor)).item()
                y_true.append(label_idx)
                y_probs.append(prob)
                processed += 1
                if processed % 10 == 0:
                    progress_bar.progress(min(processed / total_files, 1.0))
            except:
                pass

    duration = metrics_mgr.stop_timer()
    return y_true, y_probs, duration


# --- UI Layout ---
st.title("🧬 Medical Anomaly Detection Platform")

gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
st.markdown(f"**System Status:** Detected `{gpu_count}` GPU(s) ({gpu_name})")

# Look for files in CACHE_DIR
model_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pth')]

# Create 3 Tabs
tab_train, tab_infer, tab_eval = st.tabs(["🚂 Train Model", "🕵️ Single Scan Analysis", "📊 Full Evaluation"])

# === TAB 1: TRAIN NEW MODEL ===
with tab_train:
    st.header("Train a New Model")
    col1, col2 = st.columns(2)
    with col1:
        train_arch = st.selectbox("Select Architecture", ["DenseNet121", "ResNet50", "EfficientNetB0"])
        epochs = st.slider("Epochs", min_value=1, max_value=20, value=3)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    with col2:
        st.info("⚠️ Training requires significant GPU power.")
        start_btn = st.button("Start Training", type="primary")

    if start_btn:
        st.divider()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Initializing...")
            # Training now returns path inside 'cache/'
            save_path, acc, duration = train_model_pipeline(
                train_arch, epochs, batch_size, progress_bar, status_text
            )

            st.success(f"✅ Training Complete!")
            st.write(f"**Saved to:** `{save_path}`")
            st.write(f"**Final Accuracy:** {acc:.2%}")

            time.sleep(2)
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {e}")

# === TAB 2: SINGLE PREDICTION ===
with tab_infer:
    if not model_files:
        st.warning(f"No models found in '{CACHE_DIR}'. Please train a model first!")
    else:
        selected_file = st.selectbox("Select Model for Inference", model_files)

        # Construct full path: cache/model.pth
        full_model_path = os.path.join(CACHE_DIR, selected_file)

        inferred_arch = next((arch for arch in ["DenseNet121", "ResNet50", "EfficientNetB0"] if arch in selected_file),
                             "DenseNet121")

        model = load_pytorch_model(inferred_arch, full_model_path)

        uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png"])
        if uploaded_file and model:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, width=300)

            prob, lat = predict_single(model, img)

            if prob > 0.5:
                st.error(f"⚠️ **ANOMALY DETECTED** (Pneumonia) - {(prob * 100):.1f}%")
            else:
                st.success(f"✅ **NORMAL** - {((1 - prob) * 100):.1f}%")

# === TAB 3: EVALUATION ===
with tab_eval:
    if not model_files:
        st.warning("No models found.")
    else:
        st.header("Test Set Evaluation")
        eval_filename = st.selectbox("Select Model to Evaluate", model_files, key="eval_select")
        eval_path = os.path.join(CACHE_DIR, eval_filename)

        eval_arch = next((arch for arch in ["DenseNet121", "ResNet50", "EfficientNetB0"] if arch in eval_filename),
                         "DenseNet121")

        if st.button("Run Full Evaluation"):
            model_eval = load_pytorch_model(eval_arch, eval_path)
            test_dir = os.path.join('dataset', 'chest_xray', 'test')

            if os.path.exists(test_dir):
                prog = st.progress(0)
                y_true, y_probs, duration = evaluate_test_set(model_eval, test_dir, prog)
                prog.empty()

                results = metrics_mgr.calculate_metrics(y_true, y_probs)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{results['accuracy']:.2%}")
                m2.metric("Precision", f"{results['precision']:.2%}")
                m3.metric("Recall", f"{results['recall']:.2%}")
                m4.metric("F1 Score", f"{results['f1_score']:.2%}")

                fig = metrics_mgr.plot_confusion_matrix(np.array(results['confusion_matrix']))
                st.pyplot(fig)
            else:
                st.error("Test dataset not found.")