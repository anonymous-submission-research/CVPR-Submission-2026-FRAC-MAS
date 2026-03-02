import os
import logging
from typing import List, Dict, Union, Optional
from PIL import Image
import torch
from transformers import pipeline

# Configure logging
logger = logging.getLogger(__name__)

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
            # Output format check based on user snippet: output[0]["generated_text"][-1]["content"]
            # But standard pipeline might return different structure depending on version. 
            # We'll follow the user's snippet structure which assumes chat template handling.
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

            return "[Mock] MedGemma (Hosted): No API Token provided. Please set HF_TOKEN."
            
            # Attempt VQA style
            # answer = client.visual_question_answering(image=image, text=prompt)
            # return answer['answer']
            
            # Since MedGemma is text-gen, we wrap it.
            # Currently, best bet without specific endpoint knowledge is to return a placeholder 
            # or try a known working VLM pattern.
            return f"[Mock] MedGemma (Hosted): Received image ({image.size}). Prompt: {prompt}. (Real API integration requires specific Endpoint URL)."

        except Exception as e:
            logger.error(f"Error in HF Spaces prediction: {e}")
            return f"Error calling HF API: {str(e)}"
