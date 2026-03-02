# This file aggregates the necessary agent logic for the deployment environment
# to avoid complex external dependencies on 'src/' directory.

import os
import logging
import re
from typing import Dict, Any, Optional, List, Union
from PIL import Image
try:
    from backend_hf.shared import IMAGE_STORE
    # from backend_hf.app import IMAGE_STORE # (Remove this later if it exists)
except ImportError:
    try:
        from shared import IMAGE_STORE
    except ImportError:
       pass


# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# PART 1: MedGemma Client
# ==========================================

class MedGemmaClient:
    """
    Client for interacting with MedGemma VLM.
    Supports two modes:
    1. 'hf_spaces': Calls a Hugging Face Space or Inference Endpoint (Recommended).
    2. 'local': Runs the model locally using transformers pipeline (Resource intensive).
    """
    
    def __init__(self, mode: str = "hf_spaces", model_id: str = "google/medgemma-4b-it"):
        self.mode = mode or os.getenv("MEDGEMMA_MODE", "hf_spaces")
        self.model_id = model_id
        self.api_token = os.getenv("MEDGEMMA_API_TOKEN") # Or HF_TOKEN
        self.spaces_url = os.getenv("MEDGEMMA_SPACES_URL")
        
        self.pipe = None
        
        if self.mode == "local":
            self._init_local_pipeline()
        elif self.mode == "hf_spaces":
            # Lazy init handled in predict calls via requests/huggingface_hub
            if not self.api_token and not os.getenv("HF_TOKEN"):
                logger.warning("No API token found for HF Spaces. Set MEDGEMMA_API_TOKEN or HF_TOKEN.")

    def _init_local_pipeline(self):
        """Initialize local transformers pipeline."""
        try:
            logger.info(f"Initializing local MedGemma pipeline with model: {self.model_id}")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_id,
                torch_dtype=torch.bfloat16,
                device=device,
            )
            logger.info("Local pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize local pipeline: {e}")
            raise

    def predict(self, image: Image.Image, prompt: str, system_prompt: str = "You are an expert radiologist.") -> str:
        """
        Send an image and prompt to MedGemma and get the text response.
        """
        if self.mode == "local":
            return self._predict_local(image, prompt, system_prompt)
        else:
            return self._predict_hf_spaces(image, prompt, system_prompt)

    def _predict_local(self, image: Image.Image, prompt: str, system_prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        try:
            output = self.pipe(text=messages, max_new_tokens=200)
            generated_text = output[0]["generated_text"]
            if isinstance(generated_text, list):
                 return generated_text[-1]["content"]
            return generated_text
        except Exception as e:
            logger.error(f"Error in local prediction: {e}")
            return f"Error: {str(e)}"

    def _predict_hf_spaces(self, image: Image.Image, prompt: str, system_prompt: str) -> str:
        """
        Call a specific HF Space using gradio_client.
        Defaults to 'warshanks/medgemma-4b-it' if no custom space URL is set.
        """
        token = self.api_token or os.getenv("HF_TOKEN")
        # Default space if none configured
        space_id = self.spaces_url or "warshanks/medgemma-4b-it"

        try:
            from gradio_client import Client as GradioClient, handle_file
            import tempfile
            
            # Save image to temp file for Gradio
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # Initialize Gradio Client for the specific space
                logger.info(f"Connecting to HF Space: {space_id}")
                client_gradio = GradioClient(space_id, token=token)
                
                # Call the /chat endpoint as specified in documentation
                result = client_gradio.predict(
                    message={"text": prompt, "files": [handle_file(tmp_path)]},
                    param_2=system_prompt,  # System Prompt
                    param_3=512,            # Max New Tokens (conservative default)
                    api_name="/chat"
                )
                
                # Result is typically the response string directly
                logger.info(f"MedGemma Space Response: {result}")
                return str(result)
                
            except Exception as e:
                logger.error(f"Error calling Gradio Space '{space_id}': {e}")
                return f"Error from MedGemma Space: {str(e)}"
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except ImportError:
            logger.error("gradio_client not installed.")
            return "Error: gradio_client library missing. pip install gradio_client"
        except Exception as e:
            logger.error(f"Error in HF Spaces prediction: {e}")
            return f"Error: {str(e)}"

# ==========================================
# PART 2: Critic Agent
# ==========================================

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
        """
        
        prompt = self._construct_prompt(prediction_label, context_definition)
        logger.info(f"Critic Agent reviewing '{prediction_label}' with Prompt: {prompt}")
        
        response_text = self.client.predict(image, prompt)
        logger.info(f"Critic Agent response: {response_text}")
        
        parsed_result = self._parse_response(response_text)
        
        # Determine if we should flag for human review based on the critique
        flagged = parsed_result["verdict"] == "no"
        
        return {
            "critic_response_text": response_text,
            "verdict": parsed_result["verdict"], # yes, no, uncertain
            "critic_confidence": parsed_result.get("confidence", 0.0), # Estimated from text if possible
            "explanation": parsed_result.get("explanation", response_text),
            "flagged_for_human": flagged
        }

    def _construct_prompt(self, label: str, definition: str) -> str:
        """
        Constructs the prompt for the VLM.
        """
        return (
            f"The provisional diagnosis for this X-ray is '{label}'. "
            f"Reference definition: {definition} "
            f"Question: Does this image effectively demonstrate the visual features of {label}? "
            f"Answer with 'Yes' or 'No', followed by a brief explanation of the visual evidence."
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parses the crude text response from the VLM into structured data.
        """
        text_lower = text.lower().strip()
        
        verdict = "uncertain"
        if text_lower.startswith("yes"):
            verdict = "yes"
        elif text_lower.startswith("no"):
            verdict = "no"
            
        # Try to extract confidence if explicitly stated (rare in simple VLM output without CoT prompting)
        # For now, we assume high confidence if the answer is definitive
        confidence = 0.8 if verdict in ["yes", "no"] else 0.5
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": text
        }

# ==========================================
# PART 3: Consensus Utils
# ==========================================

def evaluate_consensus(
    vision_prediction: Dict[str, Any],
    critic_review: Dict[str, Any],
    delta_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Evaluates the consensus between the Vision Agent and the Critic Agent.
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
    final_decision = "approved"
    reason = "Consensus reached."
    
    if is_contradiction:
        final_decision = "flagged"
        reason = "Critic Agent contradicted the diagnosis."
    elif abs(vision_conf - critic_score) > delta_threshold:
        # If discrepancy is large (e.g. Vision 0.9, Critic 0.5), we might flag
        # But if both are high (0.8, 0.9), it's fine.
        # Here we care if one is high and other is low.
        pass

    return {
        "final_decision": final_decision,
        "vision_confidence": vision_conf,
        "critic_score": critic_score,
        "reason": reason
    }
