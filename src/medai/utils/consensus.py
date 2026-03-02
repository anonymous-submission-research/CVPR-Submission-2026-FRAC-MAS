from typing import Dict, Any

def evaluate_consensus(
    vision_prediction: Dict[str, Any],
    critic_review: Dict[str, Any],
    delta_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Evaluates the consensus between the Vision Agent and the Critic Agent.
    
    Args:
        vision_prediction: Dict with 'label' and 'confidence' (0.0-1.0).
        critic_review: Dict with 'verdict' ('yes'/'no'/'uncertain') and 'critic_confidence'.
        delta_threshold: Difference in confidence that triggers a review flag.
        
    Returns:
        Dict with:
        - final_decision: "approved" | "flagged"
        - adjusted_confidence: float
        - reason: str
    """
    
    vision_conf = vision_prediction.get("confidence", 0.0)
    critic_verdict = critic_review.get("verdict", "uncertain")
    
    # Map critic verdict to a score for comparison if not provided
    # Yes -> 0.9 (Supportive)
    # No -> 0.1 (Contradicting)
    # Uncertain -> 0.5 (Neutral)
    if "critic_confidence" in critic_review and critic_review["critic_confidence"] > 0:
         critic_score = critic_review["critic_confidence"]
         # Adjust score direction based on verdict
         if critic_verdict == "no" and critic_score > 0.5:
             # High confidence "No" means low probability of the label
             critic_score = 1.0 - critic_score
    else:
        # Fallback if no numeric confidence from critic
        if critic_verdict == "yes":
            critic_score = 0.9
        elif critic_verdict == "no":
            critic_score = 0.1
        else:
            critic_score = 0.5

    # Check for direct contradiction (Vision says X, Critic says NOT X)
    is_contradiction = (critic_verdict == "no")
    
    # Check regarding confidence gap
    # If Vision is very sure (0.9) and Critic is unsure (0.5), diff is 0.4 -> Flag
    # If Vision is sure (0.9) and Critic agrees (0.9), diff is 0.0 -> OK
    confidence_delta = abs(vision_conf - critic_score)
    
    requires_human_review = is_contradiction or (confidence_delta > delta_threshold)
    
    final_status = "flagged" if requires_human_review else "approved"
    
    # Simple adjustment: average them if flagged, or keep vision if approved?
    # Usually we penalize confidence if there is disagreement
    if requires_human_review:
        adjusted_conf = (vision_conf + critic_score) / 2
        reason = f"Critic disagreement (Delta: {confidence_delta:.2f}). Critic verdict: {critic_verdict}."
    else:
        adjusted_conf = vision_conf
        reason = "Consensus reached."

    return {
        "final_decision": final_status,
        "vision_confidence": vision_conf,
        "critic_score": critic_score,
        "adjusted_confidence": adjusted_conf,
        "reason": reason,
        "human_review_required": requires_human_review
    }
