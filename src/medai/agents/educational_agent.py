import json
from typing import Dict, Any

class EducationalAgent:
    """
    Translates technical diagnosis and explanation into simple, patient-friendly terms.
    """
    def __init__(self, doctor_name: str = "your treating doctor"):
        self.doctor_name = doctor_name
        
    def translate_to_layman_terms(self, diagnosis_result: Dict[str, Any], explanation_text: str) -> Dict[str, str]:
        """
        Generates simple summaries and next steps for the patient.

        Args:
            diagnosis_result: The dictionary output from DiagnosticAgent.
            explanation_text: The string output from ExplainabilityAgent.

        Returns:
            A dictionary containing patient-friendly summary, severity, and next steps.
        """
        
        # 1. Extract Key Findings
        fracture_detected = diagnosis_result.get("fracture_detected", False)
        predicted_class = diagnosis_result.get("predicted_class", "a specific type of injury")
        confidence = diagnosis_result.get("confidence_score", 0.0)
        
        # 2. Determine Severity in Layman Terms
        severity_map = {
            "Healthy": "None",
            "Greenstick": "Mild (The bone is cracked but not completely broken through.)",
            "Transverse": "Moderate (A clean break straight across the bone.)",
            "Oblique": "Moderate (A clean break at an angle.)",
            "Spiral": "Serious (A twisting break that spirals around the bone.)",
            "Comminuted": "Severe (The bone has broken into three or more pieces.)",
            "Oblique Displaced": "Serious (The bone is broken at an angle, and the pieces are shifted out of place.)",
            "Transverse Displaced": "Serious (The bone is broken straight across, and the pieces are shifted out of place.)",
        }
        
        layman_severity = severity_map.get(predicted_class, "We need more information on this type of break.")
        
        # 3. Simplify the Explanation
        
        # Clean up the technical explanation to remove technical jargon like 'centroid' or 'activation'
        simple_explanation = explanation_text.replace("consistent with a", "which looks like a")
        simple_explanation = simple_explanation.replace("Confidence:", "Our computer model is highly sure (")
        simple_explanation = simple_explanation.replace("The model's focus is", "The computer saw a clear sign of this")
        simple_explanation = simple_explanation.replace("distal end", "end of the bone near the hand/foot")
        simple_explanation = simple_explanation.replace("proximal end", "end of the bone near the shoulder/hip")
        simple_explanation = simple_explanation.replace("humerus", "upper arm bone")
        simple_explanation = simple_explanation.replace("radius", "lower arm bone")
        simple_explanation = simple_explanation.replace("tibia", "shin bone")
        simple_explanation = simple_explanation.replace("mild", "small")
        simple_explanation = simple_explanation.replace("strong", "very clear")
        
        
        # 4. Generate Final Summary and Next Steps
        
        if not fracture_detected or predicted_class == "Healthy":
            patient_summary = (
                f"**Great news!** Our analysis suggests your bone is **healthy** "
                f"with high confidence ({confidence:.2f}). There are no signs of a fracture."
            )
            next_steps = (
                "You can discuss your pain symptoms with your doctor, but based on this image, "
                "a fracture is highly unlikely. No immediate orthopedic action is needed."
            )
        else:
            patient_summary = (
                f"Our computer analysis strongly indicates a **break in the bone** (a fracture). "
                f"The specific type appears to be a **{predicted_class}** fracture."
            )
            
            # Combine simple explanation and confidence
            patient_summary += f"\n\n**What the computer saw:** {simple_explanation}"
            patient_summary += f".\n\n**Severity Level:** {layman_severity}"
            
            next_steps = (
                "This finding requires immediate medical follow-up. Please do the following:\n"
                f"* **Do not move** the affected area.\n"
                f"* **Immediately share these findings** with {self.doctor_name}.\n"
                f"* Your doctor will confirm the diagnosis and determine the best treatment, "
                "which may involve a cast, splint, or surgery."
            )
            
        return {
            "patient_summary": patient_summary,
            "patient_severity_assessment": layman_severity,
            "next_steps_action_plan": next_steps,
        }

# --- EXAMPLE USAGE ---

if __name__ == '__main__':
    # --- SIMULATED INPUT from Diagnostic & Explainability Agents ---
    
    # Example 1: Serious Fracture
    SIMULATED_DIAGNOSIS_1 = {
        "image_path": "fracture_image.jpg",
        "fracture_detected": True,
        "predicted_class": "Spiral",
        "severity_type": "Spiral",
        "confidence_score": 0.96,
        "uncertainty_score": 0.04,
        "all_probabilities": [0.01, 0.01, 0.01, 0.01, 0.01, 0.96, 0.01, 0.01]
    }
    SIMULATED_EXPLANATION_1 = (
        "A fracture pattern consistent with a **Spiral** type is detected (Confidence: 0.96). "
        "The model's focus is clear near the **middle region** of the humerus in the center. "
        "This is based on a distinct linear focus."
    )

    # Example 2: Healthy Bone
    SIMULATED_DIAGNOSIS_2 = {
        "image_path": "healthy_image.jpg",
        "fracture_detected": False,
        "predicted_class": "Healthy",
        "severity_type": "Healthy",
        "confidence_score": 0.99,
        "uncertainty_score": 0.01,
        "all_probabilities": [0.00, 0.00, 0.99, 0.00, 0.00, 0.00, 0.00, 0.01]
    }
    SIMULATED_EXPLANATION_2 = (
        "The bone appears **healthy** with high confidence (0.99). No fracture pattern was detected."
    )


    # --- Run Agent ---
    
    agent = EducationalAgent(doctor_name="Dr. Smith")

    # Run Example 1
    results_1 = agent.translate_to_layman_terms(SIMULATED_DIAGNOSIS_1, SIMULATED_EXPLANATION_1)
    
    print("\n--- PATIENT REPORT (FRACTURE DETECTED) ---")
    print(f"**SUMMARY:** {results_1['patient_summary']}")
    print("\n**ACTION PLAN:**")
    print(results_1['next_steps_action_plan'])
    print("-------------------------------------------\n")
    
    # Run Example 2
    results_2 = agent.translate_to_layman_terms(SIMULATED_DIAGNOSIS_2, SIMULATED_EXPLANATION_2)

    print("\n--- PATIENT REPORT (HEALTHY BONE) ---")
    print(f"**SUMMARY:** {results_2['patient_summary']}")
    print("\n**ACTION PLAN:**")
    print(results_2['next_steps_action_plan'])
    print("-------------------------------------\n")