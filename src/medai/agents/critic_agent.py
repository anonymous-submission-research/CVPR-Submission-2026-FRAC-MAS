import logging
import re
from typing import Dict, Any, Optional, List
from PIL import Image
from src.medai.agents.medgemma_client import MedGemmaClient
import json
import re

logger = logging.getLogger(__name__)

class CriticAgent:
    """
    Agent responsible for 'Cyclic Self-Correction'.
    It uses a VLM (MedGemma) to double-check the diagnosis provided by the Vision Agent.
    """
    
    def __init__(self, mode: str = "hf_spaces", model_id: str = "google/medgemma-4b-it"):
        self.client = MedGemmaClient(mode=mode, model_id=model_id)

    def review_diagnosis(
        self, 
        image: Image.Image, 
        prediction_label: str, 
        prediction_confidence: float, 
        context_definition: str
    ) -> Dict[str, Any]:
        """
        Conducts a review of the diagnosis.
        
        Args:
            image: The X-ray image.
            prediction_label: The class predicted by the Vision Agent (e.g., "Transverse Fracture").
            prediction_confidence: The confidence score from the Vision Agent.
            context_definition: Definition/visual features of the condition from the Knowledge Agent.
            
        Returns:
            Dict containing:
            - verification: "confirmed" | "rejected" | "uncertain"
            - critic_confidence: float (0.0 - 1.0)
            - explanation: Textual explanation from the Critic.
            - flagged_for_human: boolean
        """
        
        # Use a structured prompt asking the VLM to first provide an independent
        # diagnosis (top label + confidence) and then compare to the ensemble.
        prompt = self._construct_prompt(prediction_label, context_definition)
        logger.info(f"Critic Agent reviewing '{prediction_label}' with Prompt: {prompt}")

        response_text = self.client.predict(image, prompt)
        logger.info(f"Critic Agent response: {response_text}")

        parsed_result = self._parse_response(response_text)

        # Backward-compatible fields and conservative flagging heuristic:
        # - flag if VLM explicitly rejects the ensemble ('no')
        # - OR if the VLM's top independent diagnosis differs from the ensemble
        #   and the VLM reports reasonably high confidence (> 0.6)
        # - OR if the VLM responds 'uncertain'
        vlad = parsed_result.get("top_diagnosis")
        vconf = parsed_result.get("top_confidence", parsed_result.get("confidence", 0.0))
        verdict = parsed_result.get("verdict", "uncertain")

        flagged = False
        if verdict == "no" or verdict == "uncertain":
            flagged = True
        elif vlad and vlad.lower() != prediction_label.lower() and vconf >= 0.6:
            flagged = True

        return {
            "critic_response_text": response_text,
            "verdict": verdict, # yes, no, uncertain
            "critic_confidence": parsed_result.get("confidence", vconf),
            "top_diagnosis": vlad,
            "top_diagnosis_confidence": vconf,
            "explanation": parsed_result.get("explanation", response_text),
            "flagged_for_human": flagged
        }

    def _construct_prompt(self, label: str, definition: str) -> str:
        """
        Constructs the prompt for the VLM.
        """
        # Request a two-step, structured response. First, ask the model to provide
        # its own independent top diagnosis and confidence. Second, ask whether the
        # image supports the provided ensemble label. Require a JSON output where
        # possible to enable robust parsing.
        return (
            f"Step 1: Without seeing any external prediction, examine the image and "
            f"provide your top diagnosis from the following set: ['Comminuted','Greenstick','Healthy','Oblique','Oblique Displaced','Spiral','Transverse','Transverse Displaced']. "
            f"Return a JSON object: {{\"top_diagnosis\": <label>, \"top_confidence\": <0-1 float>}}. "
            f"\nStep 2: Now, the provisional diagnosis from the vision ensemble is '{label}'. "
            f"Reference definition: {definition} \nQuestion: Does this image effectively demonstrate the visual features of '{label}'? "
            f"Answer with a JSON object: {{\"verdict\": " + "\"yes|no|uncertain\", \"confidence\": <0-1 float>, \"explanation\": <text>}}. "
            f"If you cannot return JSON, then a plain textual 'Yes' or 'No' with a short explanation is acceptable."
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parses the crude text response from the VLM into structured data.
        """
        # Try to extract JSON from the response first (robust path)
        out: Dict[str, Any] = {}
        # Find first JSON-like substring
        json_match = None
        try:
            # Quick attempt: if entire response is JSON
            parsed = json.loads(text)
            json_match = parsed
        except Exception:
            # Search for a JSON block inside text
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    json_match = json.loads(m.group(0))
                except Exception:
                    json_match = None

        if isinstance(json_match, dict):
            # Normalize common keys
            out["top_diagnosis"] = json_match.get("top_diagnosis") or json_match.get("diagnosis")
            out["top_confidence"] = float(json_match.get("top_confidence", json_match.get("diagnosis_confidence", 0.0)))
            out["verdict"] = json_match.get("verdict", "uncertain")
            out["confidence"] = float(json_match.get("confidence", out.get("top_confidence", 0.0)))
            out["explanation"] = json_match.get("explanation", text)
            return out

        # Fallback: simple yes/no parsing for legacy VLM replies
        text_lower = text.lower().strip()
        verdict = "uncertain"
        if text_lower.startswith("yes"):
            verdict = "yes"
        elif text_lower.startswith("no"):
            verdict = "no"

        confidence = 0.8 if verdict in ["yes", "no"] else 0.5

        # Try to extract an explicit numeric confidence phrase like 'confidence: 0.7'
        m_conf = re.search(r"([0-9]*\.?[0-9]+)\s*(?:confidence|probability|pct|%)", text.lower())
        if m_conf:
            try:
                confidence = float(m_conf.group(1))
                if confidence > 1.0:
                    # if percent given like 75%, normalize
                    confidence = confidence / 100.0
            except Exception:
                pass

        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": text
        }
