"""
Streamlit dashboard for Wildlife Guardian.
Shows:
- Live webcam feed with detections
- Activity graphs (detections and threats)
- Recent threat log (Poacher / Endangered)
"""

from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import json

import cv2
import pandas as pd
import streamlit as st
import timm
import torch
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "yolo_model": "yolov8n.pt",
    "yolo_conf": 0.55,
    "classification_model": "resnet50.a1_in1k",
    "classification_conf": 0.15,
    "classify_every_n_frames": 8,
    "frame_width": 960,
    "frame_height": 540,
    "live_update_interval_ms": 250,
    "label_font_scale": 1.0,
    "label_thickness": 3,
    "label_line_height": 36,
    "label_padding": 8,
    "history_points": 120,
    "threat_log_size": 200,
}

ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

WILDLIFE_SPECIES_KEYWORDS = {
    "tiger": ["tiger", "bengal", "siberian", "amur"],
    "leopard": ["leopard", "panther", "jaguar", "clouded"],
    "lion": ["lion", "puma", "cougar", "mountain lion"],
    "elephant": ["elephant", "mammoth"],
    "rhino": ["rhinoceros", "rhino"],
    "panda": ["panda", "giant panda"],
    "bear": ["bear", "polar", "grizzly", "brown", "black bear", "sloth bear"],
    "wolf": ["wolf", "canis", "wild dog"],
    "gorilla": ["gorilla", "ape", "primate"],
    "zebra": ["zebra", "quagga"],
    "giraffe": ["giraffe"],
    "cheetah": ["cheetah", "acinonyx"],
    "deer": ["deer", "stag", "elk", "moose", "reindeer"],
    "bird": ["eagle", "hawk", "falcon", "crane", "stork", "owl"],
    "reptile": ["snake", "lizard", "crocodile", "alligator", "turtle"],
    "whale": ["whale", "dolphin", "porpoise", "cetacean"],
}

ENDANGERED_SPECIES = {
    "tiger": True,
    "leopard": True,
    "cheetah": True,
    "lion": True,
    "elephant": True,
    "rhino": True,
    "panda": True,
    "bear": True,
    "wolf": True,
    "gorilla": True,
    "zebra": True,
    "giraffe": True,
    "whale": True,
    "reptile": False,
    "bird": False,
    "deer": False,
}

GENERIC_ANIMAL_TOKENS = [
    "animal", "mammal", "bird", "reptile", "fish", "dog", "cat", "bear", "deer",
    "wolf", "fox", "elephant", "tiger", "lion", "leopard", "zebra", "giraffe"
]


def normalize_label_text(text: str) -> str:
    return text.replace("_", " ").replace(",", " ").lower().strip()


def display_label_text(text: str) -> str:
    primary = text.split(",")[0].strip().replace("_", " ")
    return primary.title() if primary else "Animal"


def get_species_category(label_text: str):
    normalized = normalize_label_text(label_text)
    for species_key, keywords in WILDLIFE_SPECIES_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return species_key
    return None


@st.cache_resource(show_spinner=False)
def load_resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_path = BASE_DIR / CONFIG["yolo_model"]
    detector = YOLO(str(yolo_path if yolo_path.exists() else CONFIG["yolo_model"]))
    if device == "cuda":
        detector.to("cuda")

    classifier = timm.create_model(CONFIG["classification_model"], pretrained=True)
    classifier.eval()
    if device == "cuda":
        classifier = classifier.cuda()

    data_config = timm.data.resolve_data_config(vars(classifier))
    preprocess = timm.data.create_transform(**data_config)

    labels_path = BASE_DIR / "imagenet_labels.json"
    imagenet_labels = None
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            imagenet_labels = json.load(f)

    return {
        "detector": detector,
        "classifier": classifier,
        "preprocess": preprocess,
        "labels": imagenet_labels,
        "device": device,
    }


def ensure_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=CONFIG["history_points"])
    if "threat_log" not in st.session_state:
        st.session_state.threat_log = deque(maxlen=CONFIG["threat_log_size"])
    else:
        st.session_state.threat_log = deque(
            [
                entry
                for entry in st.session_state.threat_log
                if isinstance(entry, dict)
                and (entry.get("threat") == "Poacher" or entry.get("species") == "Person/Poacher")
            ],
            maxlen=CONFIG["threat_log_size"],
        )
    if "endangered_log" not in st.session_state:
        st.session_state.endangered_log = deque(maxlen=CONFIG["threat_log_size"])
    if "wildlife_log" not in st.session_state:
        st.session_state.wildlife_log = deque(maxlen=CONFIG["threat_log_size"])
    if "species_cache" not in st.session_state:
        st.session_state.species_cache = {}
    if "species_counts" not in st.session_state:
        st.session_state.species_counts = Counter()
    if "last_frame_rgb" not in st.session_state:
        st.session_state.last_frame_rgb = None
    if "last_detections" not in st.session_state:
        st.session_state.last_detections = 0
    if "last_threats" not in st.session_state:
        st.session_state.last_threats = 0
    if "monitor_error" not in st.session_state:
        st.session_state.monitor_error = None
    if "intrusion_duration_seconds" not in st.session_state:
        st.session_state.intrusion_duration_seconds = 0.0
    if "intrusion_active_since" not in st.session_state:
        st.session_state.intrusion_active_since = None


def get_camera():
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
        cap.set(cv2.CAP_PROP_FPS, 30)
        st.session_state.cap = cap
    return cap


def release_camera():
    cap = st.session_state.cap
    if cap is not None and cap.isOpened():
        cap.release()
    st.session_state.cap = None


def classify_roi(roi, resources):
    labels = resources["labels"]
    classifier = resources["classifier"]
    preprocess = resources["preprocess"]
    device = resources["device"]

    if labels is None or roi.size == 0:
        return "Animal", None

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    input_tensor = preprocess(pil_image).unsqueeze(0)
    if device == "cuda":
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        output = classifier(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    for idx, prob in zip(top5_idx, top5_prob):
        label_idx = int(idx.item())
        prob_value = float(prob.item())
        if label_idx >= len(labels):
            continue

        raw_label = labels[label_idx]
        normalized_label = normalize_label_text(raw_label)
        category = get_species_category(raw_label)

        if category:
            return display_label_text(raw_label), category
        if prob_value >= CONFIG["classification_conf"]:
            if any(token in normalized_label for token in GENERIC_ANIMAL_TOKENS):
                return display_label_text(raw_label), category

    return "Animal", None


def draw_detection_label(frame, lines, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = CONFIG["label_font_scale"]
    thickness = CONFIG["label_thickness"]
    line_height = CONFIG["label_line_height"]
    padding = CONFIG["label_padding"]
    top_y = y - 10 - line_height * (len(lines) - 1)
    if top_y < 24:
        top_y = y + line_height

    for index, line in enumerate(lines):
        baseline_y = top_y + index * line_height
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        x0 = max(0, x - padding)
        y0 = max(0, baseline_y - text_height - padding)
        x1 = min(frame.shape[1] - 1, x + text_width + padding)
        y1 = min(frame.shape[0] - 1, baseline_y + baseline + padding)

        # Filled black label background for visibility on bright/detailed scenes.
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.putText(
            frame,
            line,
            (x, baseline_y),
            font,
            font_scale,
            color,
            thickness,
        )


def detect_frame(frame, resources):
    detector = resources["detector"]
    results = detector(frame, conf=CONFIG["yolo_conf"], verbose=False)
    annotated = frame.copy()

    detections_count = 0
    poacher_count = 0
    frame_threat_events = []
    frame_endangered_events = []
    frame_wildlife_events = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id != 0 and class_id not in ANIMAL_CLASSES:
                continue

            detections_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cache_key = f"{class_id}_{x1//50}_{y1//50}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            confidence_score = round(confidence, 3)
            confidence_pct = confidence * 100.0

            if class_id == 0:
                poacher_count += 1
                frame_threat_events.append(
                    {
                        "timestamp": timestamp,
                        "threat": "Poacher",
                        "species": "Person/Poacher",
                        "confidence": confidence_score,
                    }
                )
                color = (0, 0, 255)
                label_lines = [
                    "POACHER DETECTED",
                    f"Confidence: {confidence_pct:.2f}%",
                ]
            else:
                species_name = "Unknown"
                species_category = None
                cached = st.session_state.species_cache.get(cache_key)
                if cached:
                    species_name, species_category, cache_frame = cached
                    if st.session_state.frame_count - cache_frame > 30:
                        st.session_state.species_cache.pop(cache_key, None)
                        species_name = "Unknown"
                        species_category = None

                if (
                    species_name == "Unknown"
                    and st.session_state.frame_count % CONFIG["classify_every_n_frames"] == 0
                ):
                    roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    species_name, species_category = classify_roi(roi, resources)
                    st.session_state.species_cache[cache_key] = (
                        species_name,
                        species_category,
                        st.session_state.frame_count,
                    )

                st.session_state.species_counts[species_name] += 1
                species_category = species_category or get_species_category(species_name)

                if species_category and ENDANGERED_SPECIES.get(species_category, False):
                    frame_endangered_events.append(
                        {
                            "timestamp": timestamp,
                            "species": species_name,
                            "confidence": confidence_score,
                        }
                    )
                    color = (128, 0, 128)
                    label_lines = [
                        f"Endangered Species: {species_name}",
                        f"Confidence: {confidence_pct:.2f}%",
                    ]
                else:
                    frame_wildlife_events.append(
                        {
                            "timestamp": timestamp,
                            "species": species_name,
                            "confidence": confidence_score,
                        }
                    )
                    color = (0, 255, 0)
                    label_lines = [
                        f"Animal Detected: {species_name}",
                        f"Confidence: {confidence_pct:.2f}%",
                    ]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            draw_detection_label(annotated, label_lines, x1, y1, color)

    return (
        annotated,
        detections_count,
        poacher_count,
        frame_threat_events,
        frame_endangered_events,
        frame_wildlife_events,
    )


st.set_page_config(
    page_title="Wildlife Guardian Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_state()

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 280px;
        max-width: 280px;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] h2,
    section[data-testid="stSidebar"][aria-expanded="false"] .stRadio {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Wildlife Guardian")

controls_col1, controls_col2, controls_col3 = st.columns(3)
with controls_col1:
    if st.button("Start Monitoring", use_container_width=True):
        st.session_state.running = True
        st.session_state.monitor_error = None
with controls_col2:
    if st.button("Stop Monitoring", use_container_width=True):
        st.session_state.running = False
        st.session_state.monitor_error = None
        if st.session_state.intrusion_active_since is not None:
            st.session_state.intrusion_duration_seconds += (
                datetime.now() - st.session_state.intrusion_active_since
            ).total_seconds()
            st.session_state.intrusion_active_since = None
        release_camera()
with controls_col3:
    if st.button("Clear Logs", use_container_width=True):
        st.session_state.history.clear()
        st.session_state.threat_log.clear()
        st.session_state.endangered_log.clear()
        st.session_state.wildlife_log.clear()
        st.session_state.species_counts.clear()
        st.session_state.intrusion_duration_seconds = 0.0
        st.session_state.intrusion_active_since = None

resources = load_resources()

st.sidebar.header("Menu")
selected_menu = st.sidebar.radio(
    "Navigate",
    (
        "Live Monitoring",
        "Threat Log (Poachers Only)",
        "Endangered Species Log",
        "Wildlife Log",
    ),
)


def process_monitoring_frame():
    if not st.session_state.running:
        return

    now = datetime.now()
    cap = get_camera()
    if not cap.isOpened():
        st.session_state.monitor_error = "Camera is not available."
        st.session_state.running = False
        if st.session_state.intrusion_active_since is not None:
            st.session_state.intrusion_duration_seconds += (
                now - st.session_state.intrusion_active_since
            ).total_seconds()
            st.session_state.intrusion_active_since = None
        release_camera()
        return

    ok, frame = cap.read()
    if not ok:
        st.session_state.monitor_error = "Failed to read camera frame."
        st.session_state.running = False
        if st.session_state.intrusion_active_since is not None:
            st.session_state.intrusion_duration_seconds += (
                now - st.session_state.intrusion_active_since
            ).total_seconds()
            st.session_state.intrusion_active_since = None
        release_camera()
        return

    st.session_state.monitor_error = None
    st.session_state.frame_count += 1
    (
        annotated,
        detections,
        threats,
        threat_events,
        endangered_events,
        wildlife_events,
    ) = detect_frame(frame, resources)
    st.session_state.last_detections = detections
    st.session_state.last_threats = threats
    st.session_state.history.append(
        {
            "timestamp": now,
            "detections": detections,
            "threats": threats,
        }
    )
    if threats > 0:
        if st.session_state.intrusion_active_since is None:
            st.session_state.intrusion_active_since = now
    elif st.session_state.intrusion_active_since is not None:
        st.session_state.intrusion_duration_seconds += (
            now - st.session_state.intrusion_active_since
        ).total_seconds()
        st.session_state.intrusion_active_since = None
    for event in threat_events:
        st.session_state.threat_log.appendleft(event)
    for event in endangered_events:
        st.session_state.endangered_log.appendleft(event)
    for event in wildlife_events:
        st.session_state.wildlife_log.appendleft(event)
    st.session_state.last_frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def render_log_table(title, entries, empty_caption):
    if entries:
        df = pd.DataFrame(list(entries))
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption(empty_caption)


def render_live_monitoring_page():
    video_col, summary_col = st.columns([2, 1])
    with summary_col:
        st.subheader("System")
        st.write(f"Device: `{resources['device'].upper()}`")
        st.write(f"YOLO: `{CONFIG['yolo_model']}`")
        st.write(f"Classifier: `{CONFIG['classification_model']}`")

    if st.session_state.monitor_error:
        st.error(st.session_state.monitor_error)

    with video_col:
        if st.session_state.last_frame_rgb is not None:
            st.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
        elif st.session_state.running:
            st.warning("Waiting for camera frames...")
        else:
            st.info("Press 'Start Monitoring' to open the webcam and begin detection.")

    with summary_col:
        st.metric("Detections (current frame)", st.session_state.last_detections)
        st.metric("Threats (current frame)", st.session_state.last_threats)
        st.metric("Threat log entries", len(st.session_state.threat_log))
        if st.session_state.last_threats > 0:
            st.error(f"ALERT: {st.session_state.last_threats} poacher(s) detected.")

    st.subheader("Species Frequency (Session)")
    if st.session_state.species_counts:
        species_df = (
            pd.DataFrame(
                [{"species": key, "count": value} for key, value in st.session_state.species_counts.items()]
            )
            .sort_values("count", ascending=False)
            .head(10)
            .set_index("species")
        )
        st.bar_chart(species_df)
    else:
        st.caption("No species detected yet.")

def render_log_page(title, log_entries, empty_caption):
    st.subheader(title)
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Detections (current frame)", st.session_state.last_detections)
    with metrics_col2:
        st.metric("Poachers (current frame)", st.session_state.last_threats)
    with metrics_col3:
        st.metric("Total entries", len(log_entries))

    if st.session_state.monitor_error:
        st.error(st.session_state.monitor_error)

    render_log_table(title, log_entries, empty_caption)


def get_intrusion_duration_seconds():
    duration = st.session_state.intrusion_duration_seconds
    if st.session_state.intrusion_active_since is not None:
        duration += (datetime.now() - st.session_state.intrusion_active_since).total_seconds()
    return max(0.0, duration)


def render_top_metrics():
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Wildlife Detected", len(st.session_state.wildlife_log))
    with metric_col2:
        st.metric("Endangered Species Sightings", len(st.session_state.endangered_log))
    with metric_col3:
        st.metric("Human Intrusion Count", len(st.session_state.threat_log))
    with metric_col4:
        st.metric("Intrusion Duration (seconds)", int(get_intrusion_duration_seconds()))


def render_selected_page():
    process_monitoring_frame()
    render_top_metrics()
    if selected_menu == "Live Monitoring":
        render_live_monitoring_page()
    elif selected_menu == "Threat Log (Poachers Only)":
        render_log_page(
            "Threat Log (Poachers Only)",
            st.session_state.threat_log,
            "No threats detected yet.",
        )
    elif selected_menu == "Endangered Species Log":
        render_log_page(
            "Endangered Species Log",
            st.session_state.endangered_log,
            "No endangered species detections yet.",
        )
    else:
        render_log_page(
            "Wildlife Log",
            st.session_state.wildlife_log,
            "No wildlife detections yet.",
        )


fragment = getattr(st, "fragment", None)
if fragment:
    run_every = f"{CONFIG['live_update_interval_ms']}ms" if st.session_state.running else None

    @fragment(run_every=run_every)
    def live_dashboard_fragment():
        render_selected_page()

    live_dashboard_fragment()
else:
    st.warning("Live auto-update requires Streamlit with `st.fragment` support.")
    render_selected_page()
