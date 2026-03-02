# MedAI Inference Backend (Hugging Face Spaces)

This backend serves the custom PyTorch models and multi-agent pipeline for the MedAI Fracture Detection System using **FastAPI**. It is designed to be deployed on **Hugging Face Spaces** (Docker SDK).

## Endpoints

| Method | Path        | Description                                                                                                                                   |
| ------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/diagnose` | Upload X-ray; returns ensemble prediction, per-model probabilities, Grad-CAM overlays, conformal set, Critic verdict, and educational summary |
| `POST` | `/chat`     | Patient follow-up chat via the LangGraph Patient Interface Agent                                                                              |
| `POST` | `/report`   | Generate and download a multi-page ReportLab PDF diagnostic report                                                                            |

## Key Modules

| File                     | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| `app.py`                 | FastAPI application, model loading, ensemble inference, agent orchestration |
| `medai_agent_module.py`  | Self-contained MedGemma Critic Agent and hybrid consensus logic             |
| `patient_agent_graph.py` | LangGraph graph for the Patient Interface Agent (gemini-2.5-flash-lite)     |
| `report_generator.py`    | ReportLab multi-page PDF report generator                                   |
| `shared.py`              | Shared in-memory state (`IMAGE_STORE`, `CLASS_NAMES`)                       |

## Deployment Instructions

1. **Create a New Space** on Hugging Face with the **Docker** SDK.

2. **Upload Models** — Place `.pth` / `.pt` checkpoints in a `models/` directory (use Git LFS):
   - `best_hypercolumn_cbam_densenet169.pth`
   - `best_maxvit.pth`
   - `best_rad_dino_classifier.pth`
   - `best.pt` (YOLOv26m-cls)

3. **Set Environment Variables** in the Space settings:

   ```
   GOOGLE_API_KEY=<your-google-ai-studio-key>      # For gemini-2.5-flash-lite and gemini-2.5-pro
   HF_TOKEN=<your-huggingface-token>                # For MedGemma access
   MEDGEMMA_MODE=hf_spaces
   MEDGEMMA_SPACES_URL=<your-medgemma-space-url>
   ```

4. **Optional Artifacts** — For conformal prediction and stacking, ensure these are accessible:
   - `conformal_threshold.txt` — calibrated nonconformity threshold (from `scripts/calibration/prepare_val_and_calibrate.py`)
   - `outputs/stacker.joblib` — trained stacking pipeline (from `scripts/training/train_stacker.py`)

5. **Deploy** — Push all files to the Space. The API is available at:
   `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

## Local Testing

```bash
pip install -r requirements.txt
python app.py
# Server runs at http://localhost:7860
```

## Notes

- Grad-CAM requires `pytorch-grad-cam`; gracefully disabled if unavailable.
- The Critic Agent uses MedGemma-4B-IT via HF Spaces API by default (`MEDGEMMA_MODE=hf_spaces`). Set `MEDGEMMA_MODE=local` for on-device inference (resource-intensive).
- ChromaDB is loaded from `./chroma_db` at startup for the Knowledge Agent.
- Each `/diagnose` call stores the original image in `IMAGE_STORE` (in-memory) keyed by `inference_id` for subsequent `/chat` and `/report` calls.
