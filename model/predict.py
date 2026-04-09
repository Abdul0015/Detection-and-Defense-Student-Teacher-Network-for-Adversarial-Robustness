import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import torch.nn.functional as F

# --- 1. MODEL ARCHITECTURE SETUP ---
def get_resnet34_cifar():
    """Modifies ResNet34 as per the paper for 32x32 CIFAR-10 images."""
    model = resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def get_features(model, x):
    """Extracts intermediate features from layer1 to layer4 as per Section III-B."""
    features = []
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    
    x = model.layer1(x)
    features.append(x)
    x = model.layer2(x)
    features.append(x)
    x = model.layer3(x)
    features.append(x)
    x = model.layer4(x)
    features.append(x)
    return features, x # returns layer list and the final feature map

# --- 2. CONFIGURATION & LOADING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
THRESHOLD = 0.05  # You should set this based on your compute_threshold function

# Load Teacher (Standard Model)
teacher = get_resnet34_cifar().to(device)
# teacher.load_state_dict(torch.load("teacher_resnet34_best.pth", map_all_range=device))
teacher.load_state_dict(torch.load("teacher_resnet34_best.pth", map_location=device))
teacher.eval()

# Load Student (Robust Feature Restorer)
student = get_resnet34_cifar().to(device)
student.fc = nn.Identity() # Student backbone only
# student.load_state_dict(torch.load("student_resnet34_best.pth", map_all_range=device))
student.load_state_dict(torch.load("student_resnet34_best.pth", map_location=device))
student.eval()

# --- 3. PREDICTION FUNCTION ---
def predict(image_path):
    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201)),
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Get Teacher Output & Features
        t_feats, t_final_map = get_features(teacher, input_tensor)
        teacher_logits = teacher.fc(torch.flatten(F.adaptive_avg_pool2d(t_final_map, (1, 1)), 1))
        teacher_probs = F.softmax(teacher_logits, dim=1)
        msp, teacher_pred = torch.max(teacher_probs, dim=1)

        # 2. Get Student Features
        s_feats, s_final_map = get_features(student, input_tensor)

        # 3. Calculate Detection Score (MAE)
        layer_maes = []
        for t_f, s_f in zip(t_feats, s_feats):
            mae = torch.abs(t_f - s_f).mean().item()
            layer_maes.append(mae)
        
        max_mae = max(layer_maes)
        detection_score = max(max_mae, 1 - msp.item())

        # 4. Defense Phase: The Switch
        is_adversarial = detection_score >= THRESHOLD
        
        # Student Prediction (Feature Restoration)
        # Passing student's restored features into teacher's trained classifier head
        student_penultimate = torch.flatten(F.adaptive_avg_pool2d(s_final_map, (1, 1)), 1)
        student_logits = teacher.fc(student_penultimate)
        student_pred = torch.argmax(student_logits, dim=1)

        final_decision = student_pred if is_adversarial else teacher_pred

    # --- 4. DISPLAY PERFORMANCE & METRICS ---
    print("\n" + "="*50)
    print("      STUDENT-TEACHER DEFENSE SYSTEM RESULTS")
    print("="*50)
    print(f"Detected Status: {'⚠️ ADVERSARIAL' if is_adversarial else '✅ NORMAL'}")
    print(f"Detection Score: {detection_score:.4f} (Threshold: {THRESHOLD})")
    print(f"Teacher MSP (Confidence): {msp.item():.4f}")
    print(f"Feature Distortion (MAE): {max_mae:.4f}")
    print("-" * 50)
    
    print(f"Teacher Prediction: {CLASSES[teacher_pred]}")
    print(f"Student Restoration: {CLASSES[student_pred]}")
    print("-" * 50)
    print(f"FINAL SYSTEM OUTPUT: {CLASSES[final_decision.item()]}")
    print("-" * 50)
    
    # Comparison logic based on paper
    if is_adversarial:
        print("EXPLANATION: The input was flagged as an attack. The Teacher's ")
        print("prediction is likely corrupted. The Student Network has ")
        print("successfully restored the features for a robust prediction.")
    else:
        print("EXPLANATION: The input is a Normal Example (NE). The system ")
        print("uses the Teacher's high-precision classification to ensure ")
        print("zero performance degradation on clean data.")
    print("="*50 + "\n")

    final_label = CLASSES[final_decision.item()]

    return {
        "is_adversarial": is_adversarial,
        "detection_score": round(detection_score, 4),
        "confidence": round(msp.item(), 4),
        "mae": round(max_mae, 4),
        "teacher_pred": CLASSES[teacher_pred.item()],
        "student_pred": CLASSES[student_pred.item()],
        "final_output": final_label
    }

if __name__ == "__main__":
    path = r"cifar10_test_images\airplane\111.png"
    predict(path)