import numpy as np
import json
from typing import Dict, Any

# --- New Helper Function for Dynamic Testing ---
def generate_random_heatmap(size: int = 224) -> np.ndarray:
    """
    Generates a randomized, plausible heatmap array for testing the agent's dynamism.
    The heatmap will have a focused, high-intensity area somewhere random.
    """
    # Create a base array of zeros
    cam_array = np.zeros((size, size), dtype=np.float32)
    
    # 1. Define random center and size for the activation zone
    center_y = np.random.randint(size // 4, size * 3 // 4)
    center_x = np.random.randint(size // 4, size * 3 // 4)
    height = np.random.randint(30, 80)
    width = np.random.randint(30, 80)
    
    # Define activation bounds (ensure they stay within the array limits)
    y_min = max(0, center_y - height // 2)
    y_max = min(size, center_y + height // 2)
    x_min = max(0, center_x - width // 2)
    x_max = min(size, center_x + width // 2)
    
    # 2. Apply activation with random strength
    random_strength = np.random.uniform(0.6, 1.0)
    cam_array[y_min:y_max, x_min:x_max] = random_strength
    
    # Optional: Add minor noise to make it less blocky
    cam_array = cam_array + np.random.uniform(0, 0.1, (size, size))
    cam_array = np.clip(cam_array, 0, 1)

    return cam_array

# --- Helper function for localization (No changes needed, it is dynamic) ---

def calculate_heatmap_centroid(cam_array: np.ndarray, threshold: float = 0.5) -> tuple:
    """
    Calculates the centroid (center of mass) of the significant activation area
    in the Grad-CAM heatmap.
    """
    # 1. Apply threshold to isolate the 'hot' region
    binary_map = cam_array > threshold
    
    if not np.any(binary_map):
        return (0.5, 0.5, 0.0)

    # 2. Calculate coordinates and weights (activation values)
    coords = np.argwhere(binary_map)
    weights = cam_array[binary_map]

    if len(weights) == 0:
        return (0.5, 0.5, 0.0)

    # 3. Calculate weighted average for the centroid
    y_coords = coords[:, 0] # Rows (Y)
    x_coords = coords[:, 1] # Columns (X)

    sum_weights = np.sum(weights)
    
    centroid_x = np.sum(x_coords * weights) / sum_weights
    centroid_y = np.sum(y_coords * weights) / sum_weights

    # Normalize to [0, 1] based on map size
    h, w = cam_array.shape
    norm_x = centroid_x / w
    norm_y = centroid_y / h

    max_activation = np.max(weights)

    return (norm_x, norm_y, max_activation)

# --- Explainability Agent Core (No changes needed, logic is dynamic) ---

class ExplainabilityAgent:
    def __init__(self, class_names: list, body_part: str = "bone"):
        self.class_names = class_names
        self.body_part = body_part

    def generate_explanation(self, diagnosis_result: Dict[str, Any], cam_array: np.ndarray) -> str:
        """
        Converts the Grad-CAM heatmap and prediction result into a textual explanation.
        """
        predicted_class = diagnosis_result.get("predicted_class", "Unknown")
        confidence = diagnosis_result.get("confidence_score", 0.0)
        
        # 1. Analyze Heatmap
        norm_x, norm_y, strength = calculate_heatmap_centroid(cam_array, threshold=0.4)
        
        # Determine general location (Simplified)
        x_loc = "right side" if norm_x > 0.65 else ("left side" if norm_x < 0.35 else "center")
        y_loc = "distal end" if norm_y > 0.65 else ("proximal end" if norm_y < 0.35 else "middle region")
        
        # 2. Build Textual Explanation based on Prediction
        
        if predicted_class == "Healthy":
            if confidence > 0.90:
                return f"The {self.body_part} appears **healthy** with high confidence ({confidence:.2f}). No fracture pattern was detected."
            else:
                return f"The {self.body_part} is likely **healthy** ({confidence:.2f}), though there is some low activation in the {y_loc} of the {x_loc} that warrants a closer look."

        if not diagnosis_result.get("fracture_detected", True): # Default to True if key missing
             return f"Diagnosis is **inconclusive** or data is missing."
             
        # 3. Explanation for Detected Fracture
        
        intro = f"A fracture pattern consistent with a **{predicted_class}** type is detected"
        
        # Strength adjective
        if strength > 0.7:
            strength_adj = "strong"
        elif strength > 0.5:
            strength_adj = "clear"
        else:
            strength_adj = "mild"
            
        # Confidence statement
        confidence_stmt = f"(Confidence: {confidence:.2f})"
        
        # Location statement
        location_stmt = f"near the **{y_loc}** of the {self.body_part} in the {x_loc}."
        
        # Final Assembly
        explanation = f"{intro} {confidence_stmt}. The model's focus is {strength_adj} {location_stmt}"
        
        # Add a note on the type based on visual focus
        if predicted_class in ["Transverse", "Oblique"]:
            explanation += " This is based on a distinct linear focus."
        
        return explanation


# --- 4. EXAMPLE USAGE ---

if __name__ == '__main__':
    
    # --- SIMULATED INPUT ---
    SIMULATED_RESULT = {
        "image_path": "test_image.jpg",
        "fracture_detected": True,
        "predicted_class": "Spiral",
        "severity_type": "Spiral",
        "confidence_score": 0.95,
        "uncertainty_score": 0.05,
    }

    CLASS_NAMES = ["Comminuted", "Greenstick", "Healthy", "Oblique", "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
    explainer = ExplainabilityAgent(class_names=CLASS_NAMES, body_part="humerus")
    
    # Run 3 times to demonstrate dynamic output
    print("\n--- Testing Dynamic Output (Run 1: Random Heatmap) ---")
    
    # Use the new dynamic heatmap function!
    dynamic_cam_1 = generate_random_heatmap()
    explanation_text_1 = explainer.generate_explanation(SIMULATED_RESULT, dynamic_cam_1)
    print(f"Explanation 1: {explanation_text_1}")

    print("\n--- Testing Dynamic Output (Run 2: Another Random Heatmap) ---")
    dynamic_cam_2 = generate_random_heatmap()
    explanation_text_2 = explainer.generate_explanation(SIMULATED_RESULT, dynamic_cam_2)
    print(f"Explanation 2: {explanation_text_2}")
    
    print("--------------------------------------------------\n")