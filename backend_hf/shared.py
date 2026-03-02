
# Shared in-memory storage and constants to avoid circular imports and duplicate module loading
# ensuring all parts of the application access the same state.

import logging

# Configure logger
logger = logging.getLogger(__name__)

# Global Image Store for Chat Agent (In-Memory for Demo)
# In production, use Redis or S3/Blob storage
IMAGE_STORE = {}

# We can also re-export or define common constants here if needed by multiple modules
CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique", 
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
