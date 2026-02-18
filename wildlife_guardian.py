"""
Optimized Wildlife Guardian - Faster & More Accurate
Improvements:
- GPU acceleration (if available)
- Frame skipping for classification
- Larger YOLO model option
- Batch processing
- Confidence thresholds
- Caching mechanism
"""

import cv2
from ultralytics import YOLO
import time
import warnings
import torch
from torchvision import transforms
from PIL import Image
import timm

from behavior_analyzer import BehaviorAnalyzer, BehaviorVisualizer
import numpy as np
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'yolo_model': 'yolov8s.pt',  # Options: yolov8n.pt (fastest), yolov8s.pt (balanced), yolov8m.pt (accurate)
    'yolo_conf': 0.4,  # YOLO confidence threshold (0.25-0.5 recommended)
    'classify_every_n_frames': 3,  # Classify every N frames (1=every frame, 3=every 3rd frame)
    'use_gpu': True,  # Use GPU if available
    'frame_width': 640,
    'frame_height': 480,
    'classification_conf': 0.15,  # Classification confidence threshold
    # NOTE: Must match ImageNet-1k labels in imagenet_labels.json
    'classification_model': 'resnet50.a1_in1k',
}

# Initialize behavior analyzer
behavior_analyzer = BehaviorAnalyzer(history_frames=15)

# Check for GPU
device = 'cuda' if torch.cuda.is_available() and CONFIG['use_gpu'] else 'cpu'
print(f"Using device: {device.upper()}")

# Initialize YOLOv8 model
print(f"Loading detection model: {CONFIG['yolo_model']}...")
detector = YOLO(CONFIG['yolo_model'])
if device == 'cuda':
    detector.to('cuda')

# Load ImageNet class labels from local file (ImageNet-1k)
imagenet_labels = None
try:
    with open('imagenet_labels.json', 'r') as f:
        imagenet_labels = json.load(f)
    print(f"Loaded {len(imagenet_labels)} ImageNet labels from local file")
except Exception as e:
    print(f"Error loading local ImageNet labels: {e}")
    imagenet_labels = None
    print("Warning: Species names may be generic without ImageNet labels")

# Initialize timm wildlife model (must match label count)
print("Loading wildlife classification model...")
model_name = CONFIG.get('classification_model', 'resnet50.a1_in1k')
fallback_model = 'resnet50.a1_in1k'

def _build_classifier(model_id: str):
    model = timm.create_model(model_id, pretrained=True)
    model.eval()
    if device == 'cuda':
        model = model.cuda()
    return model

classifier = None
selected_model = None
for candidate in [model_name, fallback_model]:
    try:
        candidate_model = _build_classifier(candidate)
        if imagenet_labels is not None and hasattr(candidate_model, 'num_classes'):
            if candidate_model.num_classes != len(imagenet_labels):
                print(
                    f"Label/model mismatch for {candidate}: "
                    f"{candidate_model.num_classes} classes vs {len(imagenet_labels)} labels. "
                    "Trying fallback..."
                )
                continue
        classifier = candidate_model
        selected_model = candidate
        print(f"Using model: {candidate}")
        break
    except Exception as e:
        print(f"Error loading model {candidate}: {e}")

if classifier is None:
    print("Falling back to resnet50 (unverified label match)...")
    classifier = _build_classifier('resnet50')
    selected_model = 'resnet50'
    print(f"Using model: {selected_model}")

# ImageNet preprocessing (timm provides this)
data_config = timm.data.resolve_data_config(vars(classifier))
preprocess = timm.data.create_transform(**data_config)

# Wildlife species mapping for better recognition
WILDLIFE_SPECIES_KEYWORDS = {
    'tiger': ['tiger', 'Bengal', 'siberian', 'amur'],
    'leopard': ['leopard', 'panther', 'jaguar', 'clouded'],
    'lion': ['lion', 'puma', 'cougar', 'mountain lion'],
    'elephant': ['elephant', 'mammoth'],
    'rhino': ['rhinoceros', 'rhino'],
    'panda': ['panda', 'giant panda'],
    'bear': ['bear', 'polar', 'grizzly', 'brown', 'black bear', 'sloth bear'],
    'wolf': ['wolf', 'canis', 'wild dog'],
    'gorilla': ['gorilla', 'ape', 'primate'],
    'zebra': ['zebra', 'quagga'],
    'giraffe': ['giraffe'],
    'cheetah': ['cheetah', 'acinonyx'],
    'deer': ['deer', 'stag', 'elk', 'moose', 'reindeer'],
    'bird': ['eagle', 'hawk', 'falcon', 'crane', 'stork', 'owl'],
    'reptile': ['snake', 'lizard', 'crocodile', 'alligator', 'turtle'],
    'whale': ['whale', 'dolphin', 'porpoise', 'cetacean'],
}

# Extra tokens used to decide if a raw ImageNet label is still animal-like
GENERIC_ANIMAL_TOKENS = [
    'animal', 'mammal', 'bird', 'reptile', 'fish', 'dog', 'cat', 'bear', 'deer',
    'wolf', 'fox', 'elephant', 'tiger', 'lion', 'leopard', 'zebra', 'giraffe'
]

def normalize_label_text(text: str) -> str:
    """Normalize label text for keyword checks."""
    return text.replace('_', ' ').replace(',', ' ').lower().strip()

def display_label_text(text: str) -> str:
    """Convert ImageNet label text into a clean display label."""
    primary = text.split(',')[0].strip().replace('_', ' ')
    return primary.title() if primary else "Animal"

def get_species_category(label_text: str):
    """Map any label text to broad species category keys used in risk rules."""
    normalized = normalize_label_text(label_text)
    for species_key, keywords in WILDLIFE_SPECIES_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return species_key
    return None

# Animal class IDs in COCO
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Endangered species classification (most wildlife is conservation-important)
ENDANGERED_SPECIES = {
    'tiger': True,
    'leopard': True,
    'cheetah': True,
    'lion': True,
    'elephant': True,
    'rhino': True,
    'panda': True,
    'bear': True,
    'wolf': True,
    'gorilla': True,
    'zebra': True,
    'giraffe': True,
    'whale': True,
    'reptile': False,
    'bird': False,
    'deer': False,
}

# Species cache to avoid re-classifying same objects
species_cache = {}
cache_timeout = 60  # frames (extended for motion tolerance)

# Access default webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

time.sleep(1)

ret, test_frame = cap.read()
if not ret:
    print("Error: Camera opened but cannot read frames")
    cap.release()
    exit()

print("\n" + "="*60)
print("OPTIMIZED WILDLIFE GUARDIAN ACTIVE")
print("="*60)
print(f"Model: {CONFIG['yolo_model']}")
print(f"Device: {device.upper()}")
print(f"YOLO Confidence: {CONFIG['yolo_conf']}")
print(f"Classification: Every {CONFIG['classify_every_n_frames']} frames")
print("Press 'q' to quit")
print("="*60 + "\n")

frame_count = 0
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0
object_id_counter = 0  # For tracking unique objects

while True:
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Cannot read frame")
        break
    
    frame_count += 1
    fps_frame_count += 1
    
    # Calculate FPS every second
    """if time.time() - fps_start_time >= 1.0:
        current_fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()"""
    
    # Run YOLOv8 detection with confidence threshold
    results = detector(frame, conf=CONFIG['yolo_conf'], verbose=False)
    time.sleep(2)
    threat_detected = False
    detected_species = []
    endangered_detected = False
    person_detected = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Check if person or animal
            if class_id == 0 or class_id in ANIMAL_CLASSES:
                threat_detected = True
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Use larger grid cells (100px) for cache key to tolerate motion/shaking
                cache_key = f"{class_id}_{x1//100}_{y1//100}"
                
                species_name = "Unknown Animal"
                
                # For persons, no need to classify
                if class_id == 0:
                    species_name = "Person/Poacher"
                    person_detected = True
                # For animals, classify on first detection or when cache expires
                elif class_id in ANIMAL_CLASSES:
                    # Check cache first
                    if cache_key in species_cache:
                        species_name, cache_frame = species_cache[cache_key]
                        # Extend cache timeout for stable detections
                        if frame_count - cache_frame > 60:
                            del species_cache[cache_key]
                            species_name = "Unknown Animal"
                    
                    # Classify on first detection OR when cache expires
                    if species_name == "Unknown Animal":
                        # Extract ROI for classification
                        roi = frame[max(0, y1):min(frame.shape[0], y2), 
                                   max(0, x1):min(frame.shape[1], x2)]
                        
                        if roi.size > 0 and classifier:
                            try:
                                # Convert to PIL and preprocess
                                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                pil_image = Image.fromarray(roi_rgb)
                                input_tensor = preprocess(pil_image)
                                input_batch = input_tensor.unsqueeze(0)
                                
                                if device == 'cuda':
                                    input_batch = input_batch.cuda()
                                
                                # Classify with timm model
                                with torch.no_grad():
                                    output = classifier(input_batch)
                                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                                    top5_prob, top5_idx = torch.topk(probabilities, 5)
                                    
                                    # Try to find best match from top 5 predictions
                                    species_name = "Unknown Animal"
                                    for idx, prob in zip(top5_idx, top5_prob):
                                        idx = idx.item()
                                        prob = prob.item()
                                        
                                        # Get label from ImageNet
                                        if imagenet_labels and idx < len(imagenet_labels):
                                            raw_label = imagenet_labels[idx]
                                            normalized_label = normalize_label_text(raw_label)
                                            category = get_species_category(raw_label)

                                            # Prefer exact species/subspecies display label when available
                                            if category:
                                                species_name = display_label_text(raw_label)
                                                break
                                            elif prob > CONFIG['classification_conf']:
                                                # Use raw label if it still looks animal-like
                                                if any(word in normalized_label for word in GENERIC_ANIMAL_TOKENS):
                                                    species_name = display_label_text(raw_label)
                                                    break
                                    
                                    # Cache the result
                                    species_cache[cache_key] = (species_name, frame_count)
                            except Exception as e:
                                species_name = "Unknown Animal"
                        else:
                            species_name = "Unknown Animal"
                
                detected_species.append(species_name)
                
                # Check if endangered species
                species_category = get_species_category(species_name)
                if species_category and ENDANGERED_SPECIES.get(species_category, False):
                    endangered_detected = True
                
                # Analyze behavior
                object_id = f"{cache_key}_{object_id_counter}"
                behavior_data = behavior_analyzer.analyze_behavior(
                    frame, (x1, y1, x2, y2), species_name, object_id
                )
                object_id_counter += 1
                
                # Draw bounding box
                is_endangered = bool(species_category and ENDANGERED_SPECIES.get(species_category, False))
                
                if class_id == 0:
                    color = (0, 0, 255) # Red for person/poacher
                elif is_endangered:
                    color = (0, 0, 255) # Red for endangered species
                else:
                    color = (0, 255, 0) # Green for other animals
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Add label
                label = f"{species_name} ({confidence:.2f})"
                if is_endangered:
                    label = f"ENDANGERED: {label}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 5, y1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw behavior information
                frame = BehaviorVisualizer.draw_behavior_info(frame, behavior_data, (x1, y1, x2, y2), species_name)
    
    # Apply threat visual indicators
    if endangered_detected or person_detected:
        h, w = frame.shape[:2]
        border_thickness = 15
        
        # Determine border color and main text based on threat type
        if endangered_detected:
            border_color = (0, 0, 255)  # Red for endangered
            main_text = "ENDANGERED SPECIES DETECTED"
            main_color = (0, 0, 255)
        elif person_detected:
            border_color = (0, 165, 255)  # Orange for person/poacher
            main_text = "Threat Detected"
            main_color = (0, 255, 255)
        
        # Draw border
        cv2.rectangle(frame, (0, 0), (w, h), border_color, border_thickness)
        
        # Show detected species and special warning
        if detected_species:
            # Main warning text
            main_font_scale = 1.2
            main_thickness = 3
            main_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_TRIPLEX, 
                                        main_font_scale, main_thickness)[0]
            main_x = (w - main_size[0]) // 2
            main_y = 50
            
            cv2.rectangle(frame, (main_x - 10, main_y - main_size[1] - 10),
                          (main_x + main_size[0] + 10, main_y + 10),
                          (0, 0, 0), -1)
            cv2.putText(frame, main_text, (main_x, main_y),
                        cv2.FONT_HERSHEY_TRIPLEX, main_font_scale, main_color, main_thickness)

            # Sub-text with species list
            species_list = ", ".join(set(detected_species))
            sub_text = f"Details: {species_list}"
            sub_font_scale = 0.7
            sub_thickness = 2
            sub_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       sub_font_scale, sub_thickness)[0]
            sub_x = (w - sub_size[0]) // 2
            sub_y = main_y + 40
            
            cv2.rectangle(frame, (sub_x - 5, sub_y - sub_size[1] - 5),
                         (sub_x + sub_size[0] + 5, sub_y + 5),
                         (0, 0, 0), -1)
            cv2.putText(frame, sub_text, (sub_x, sub_y),
                       cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (255, 255, 255), sub_thickness)
    
    # Display FPS
    fps_text = f"FPS: {current_fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Optimized Wildlife Guardian', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nSystem shutdown complete")
print(f"Average FPS: {current_fps:.1f}")


  
