# knowledge_agent.py
# --------------------------------------------------------------------------
# MedAI Knowledge Agent: ChromaDB RAG + Structured Fracture Knowledge
# --------------------------------------------------------------------------
from typing import Dict, Any, List, Optional
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import chromadb
from chromadb.utils import embedding_functions
import httpx


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CHROMA_DB_PATH = "./chroma_db"
DIAG_COLLECTION_NAME = "medical_diagnoses"
SOURCE_COLLECTION_NAME = "medai_sources"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3

# LLaMA 3 / OpenAI-compatible API config
LLAMA_API_BASE = os.getenv("LLAMA_API_BASE", "")  # e.g. "http://localhost:11434/v1"
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "dummy-key")
LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "llama3")


# --------------------------------------------------------------------------
# Structured Domain Knowledge: Fracture Types
# --------------------------------------------------------------------------
MEDICAL_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Comminuted": {
        "definition": "A fracture where the bone is broken into three or more fragments.",
        "icd_code": "S52.5",
        "severity": "High",
        "treatment_guidelines": [
            "Usually requires surgical intervention (open reduction internal fixation / ORIF).",
            "Long immobilization (8-12 weeks).",
            "Requires structured physical therapy after immobilization."
        ],
        "prognosis_notes": "Risk of non-union and malunion is higher. Full recovery may take 6+ months."
    },
    "Oblique Displaced": {
        "definition": "A diagonal break where the bone fragments are separated and misaligned.",
        "icd_code": "S52.9",
        "severity": "Medium-High",
        "treatment_guidelines": [
            "Requires reduction (closed or open) to restore alignment.",
            "Often treated with casting; unstable fractures may need internal fixation."
        ],
        "prognosis_notes": "Good prognosis if reduced early and adequately stabilized."
    },
    "Healthy": {
        "definition": "No radiographic evidence of fracture.",
        "icd_code": "Z00.0",
        "severity": "Low",
        "treatment_guidelines": [
            "No specific fracture treatment required.",
            "Advise routine follow-up and monitoring of symptoms."
        ],
        "prognosis_notes": "Normal bone health based on the available imaging."
    },
    "Transverse": {
        "definition": "A fracture line that is approximately perpendicular to the long axis of the bone.",
        "icd_code": "S52.0",
        "severity": "Medium",
        "treatment_guidelines": [
            "Closed reduction and casting are common for stable fractures.",
            "Unstable patterns may require pins, screws, or plates."
        ],
        "prognosis_notes": "Generally heals well with proper immobilization and alignment."
    },
    "Spiral": {
        "definition": "A fracture caused by a twisting force, with a spiral or helical fracture line.",
        "icd_code": "S52.7",
        "severity": "Medium-High",
        "treatment_guidelines": [
            "Often requires surgical fixation due to rotational instability.",
            "Longer recovery because of associated soft-tissue injury risk."
        ],
        "prognosis_notes": "Healing can be slow; higher risk of displacement during healing."
    },
    "Greenstick": {
        "definition": "An incomplete fracture where one cortex is broken and the other is bent, typically in children.",
        "icd_code": "S52.8",
        "severity": "Low",
        "treatment_guidelines": [
            "Usually treated with simple casting or splinting.",
            "Follow-up radiographs to ensure remodeling in growing bone."
        ],
        "prognosis_notes": "Excellent prognosis; children typically heal rapidly with complete remodeling."
    },
    "Impacted": {
        "definition": "A fracture where the ends of the bone are driven into each other, shortening the bone.",
        "icd_code": "S52.2",
        "severity": "Medium",
        "treatment_guidelines": [
            "May be stable enough for casting or functional bracing.",
            "Monitor for limb shortening or joint incongruity."
        ],
        "prognosis_notes": "Generally good stability and satisfactory healing if alignment is acceptable."
    },
    "Pathologic": {
        "definition": "A fracture occurring in bone weakened by disease (e.g., osteoporosis, tumor, metastasis).",
        "icd_code": "M84.4",
        "severity": "Often High due to the underlying pathology",
        "treatment_guidelines": [
            "Treat both the fracture and the underlying disease.",
            "May require specialized surgical fixation and oncology input."
        ],
        "prognosis_notes": "Highly dependent on the underlying condition and systemic disease control."
    }
}


# --------------------------------------------------------------------------
# RAG Knowledge Base: MedAI Domain + Technical Sources
# (Condensed from your table into documents that we can embed)
# --------------------------------------------------------------------------
RAG_SOURCE_DOCS: List[Dict[str, Any]] = [
    # ----------------- Domain Knowledge (Clinical & Radiology) -----------------
    {
        "id": "ao_ota_fracture_classification",
        "category": "Fracture Classification & Terminology",
        "title": "AO/OTA Fracture Classification System",
        "content": (
            "The AO/OTA fracture classification system is the international standard for "
            "describing fractures using bone, segment and morphology codes (e.g., 31-A2). "
            "It provides precise terminology for fracture location and pattern, enabling "
            "consistent reporting and communication between clinicians. In MedAI, this "
            "serves as the core diagnostic explainer that maps model outputs to standard "
            "orthopedic language when describing why a fracture is classified a certain way."
        ),
        "use_case": "Explain exact fracture code and terminology for model-predicted fracture classes."
    },
    {
        "id": "salter_harris_classification",
        "category": "Fracture Classification & Terminology",
        "title": "Salter-Harris Classification for Pediatric Physeal Injuries",
        "content": (
            "The Salter-Harris classification describes fractures involving the epiphyseal "
            "growth plate in children (Types I–V). It guides prognosis and treatment decisions "
            "in pediatric fractures. In MedAI, this knowledge is used when the pipeline detects "
            "a probable pediatric case, allowing LLaMA 3 to give age-appropriate explanations "
            "and warn about growth plate involvement."
        ),
        "use_case": "Provide pediatric-specific explanations when the patient is a child or adolescent."
    },
    {
        "id": "aaos_orthoinfo",
        "category": "Clinical Context & Management",
        "title": "OrthoInfo (AAOS) Patient-Friendly Fracture Articles",
        "content": (
            "OrthoInfo from the American Academy of Orthopaedic Surgeons (AAOS) provides "
            "patient-friendly explanations for fractures such as distal radius, tibial shaft, "
            "and ankle fractures. The content covers symptoms, mechanism of injury, typical "
            "treatment pathways, recovery timelines and self-care advice. In MedAI, these texts "
            "inform the patient-facing interface so that explanations are understandable and "
            "aligned with standard patient education material."
        ),
        "use_case": "Generate simple, patient-facing explanations about symptoms, treatment and recovery."
    },
    {
        "id": "rockwood_green_fractures_textbook",
        "category": "Clinical Context & Management",
        "title": "Rockwood and Green's Fractures in Adults and Children",
        "content": (
            "Rockwood and Green's is a standard orthopedic reference textbook that describes "
            "diagnosis, classification, indications for surgery and complications for fractures "
            "throughout the body. In MedAI, key diagnostic and management sections are used as "
            "high-authority clinical grounding to differentiate fracture types and to reason "
            "about complications such as non-union, malunion, and neurovascular injury."
        ),
        "use_case": "Deep clinical validation and high-authority grounding for clinician-level questions."
    },
    {
        "id": "radiopaedia_fracture_entries",
        "category": "Radiology & Interpretation",
        "title": "Radiopaedia Fracture Imaging Patterns",
        "content": (
            "Radiopaedia.org hosts detailed fracture entries with example radiographs and CT scans, "
            "describing typical imaging appearances, variants and pitfalls. It explains features such "
            "as butterfly fragments, wedge patterns, cortical step-offs and subtle trabecular changes. "
            "In MedAI, this material is used to contextualize Grad-CAM heatmaps and explain which visual "
            "features the vision transformers are expected to focus on for each fracture pattern."
        ),
        "use_case": "Explain Grad-CAM regions and image features underlying the model's decision."
    },
    {
        "id": "acr_appropriateness_criteria",
        "category": "Radiology & Interpretation",
        "title": "ACR Appropriateness Criteria for Musculoskeletal Imaging",
        "content": (
            "The American College of Radiology (ACR) Appropriateness Criteria provide evidence-based "
            "recommendations on when to order additional imaging such as CT, MRI or ultrasound. For "
            "fractures, they describe indications for follow-up imaging in occult injury, complex "
            "articular involvement and postoperative assessment. MedAI uses these guidelines to suggest "
            "standard next-step imaging options in an informational (non-prescriptive) manner."
        ),
        "use_case": "Inform non-binding recommendations about when additional imaging might be considered."
    },
    {
        "id": "ai_ethics_regulation",
        "category": "Ethical & Regulatory",
        "title": "FDA AI/ML Guidelines and Health Informatics Ethics (HIMSS/AMIA)",
        "content": (
            "Regulatory and ethics documents from bodies such as the FDA, HIMSS and AMIA emphasize "
            "transparency, bias mitigation, clinical oversight and safety for AI-based medical devices. "
            "Key themes include not replacing clinician judgment, providing understandable explanations, "
            "and clearly stating limitations. MedAI uses this knowledge to ensure that the LLaMA 3 "
            "interface gives appropriate disclaimers and avoids specific patient-tailored medical advice."
        ),
        "use_case": "Generate safety disclaimers and keep explanations informational rather than prescriptive."
    },

    # ----------------- Technical & Explainability Knowledge -----------------
    {
        "id": "swin_transformer_paper",
        "category": "Model Architecture & Vision Transformers",
        "title": "Swin Transformer Architecture",
        "content": (
            "The Swin Transformer is a hierarchical vision transformer that uses shifted windows to "
            "efficiently model local and global image context. It processes images as non-overlapping "
            "patches, applies self-attention within windows and gradually builds multi-scale feature maps. "
            "In MedAI, Swin-based models serve as core vision backbones, explaining how X-ray images are "
            "tokenized and how local fracture cues and global alignment are captured."
        ),
        "use_case": "Answer technical questions about why Swin was chosen and how it processes X-ray patches."
    },
    {
        "id": "convnext_paper",
        "category": "Model Architecture & Vision Transformers",
        "title": "ConvNeXt: Modernized CNN Architecture",
        "content": (
            "ConvNeXt is a convolutional neural network architecture that modernizes ResNet-style designs "
            "to achieve transformer-level performance while retaining convolutional inductive biases. "
            "It uses large kernels, depthwise convolutions and LayerNorm to improve accuracy and efficiency. "
            "In MedAI, ConvNeXt complements Swin as an alternative backbone in the ensemble, providing "
            "architectural diversity and robustness."
        ),
        "use_case": "Explain why a CNN-style backbone is included and how it differs from Swin."
    },
    {
        "id": "grad_cam_paper",
        "category": "Explainable AI",
        "title": "Grad-CAM: Visual Explanations from Deep Networks",
        "content": (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) produces heatmaps by backpropagating "
            "gradients from a target class to convolutional feature maps, highlighting spatial regions that "
            "contribute most to the prediction. In MedAI, Grad-CAM is applied to vision transformer and "
            "ConvNeXt feature maps to produce clinically interpretable overlays on X-ray images, explaining "
            "which bone regions influenced the predicted fracture class. Limitations include coarse "
            "localization and dependence on the chosen layer."
        ),
        "use_case": "Explain how the heatmaps are generated and discuss strengths and limitations of Grad-CAM."
    },
    {
        "id": "ensemble_learning_review",
        "category": "Explainable AI",
        "title": "Ensemble Learning and Cross-Validation in MedAI",
        "content": (
            "Ensemble learning combines multiple models to improve robustness and generalization. Common "
            "strategies include majority voting, averaging of probabilities and stacking. Cross-validation "
            "quantifies performance stability across folds. In MedAI, five specialized diagnostic agents "
            "and cross-validated models are ensembled to achieve macro-F1 > 0.92, while still allowing "
            "interpretation at the level of individual agent predictions and Grad-CAM maps."
        ),
        "use_case": "Justify the ensemble agent design and answer questions about why multiple models are used."
    },
    {
        "id": "llama3_technical_report",
        "category": "Multi-Agent & RAG/LLM",
        "title": "LLaMA 3 Capabilities and Constraints",
        "content": (
            "LLaMA 3 is a large language model designed for instruction following and multi-turn dialogue. "
            "It is powerful at generating natural language explanations but may hallucinate if not grounded "
            "in external knowledge. In MedAI, LLaMA 3 is used strictly as a controlled natural language "
            "interface, grounded via retrieval-augmented generation (RAG) over curated medical and technical "
            "sources. Prompts emphasize not giving direct medical advice and staying within retrieved context."
        ),
        "use_case": "Explain how the language agent works, its limitations, and why RAG is necessary."
    },
    {
        "id": "rag_and_multi_agent_frameworks",
        "category": "Multi-Agent & RAG/LLM",
        "title": "RAG and Multi-Agent Framework Concepts",
        "content": (
            "RAG (retrieval-augmented generation) systems combine vector search over knowledge bases with "
            "LLM generation, passing retrieved documents as context to reduce hallucinations. Multi-agent "
            "frameworks such as LangChain or CrewAI decompose complex tasks into specialized agents for "
            "data retrieval, reasoning, explanation and tool use. MedAI adopts a multi-agent architecture "
            "with dedicated diagnostic, cross-validation, explanation, patient-facing and knowledge agents, "
            "each with clearly defined responsibilities."
        ),
        "use_case": "Describe the overall MedAI multi-agent architecture and how RAG fits into it."
    }
]


# --------------------------------------------------------------------------
# Knowledge Agent Class
# --------------------------------------------------------------------------
class KnowledgeAgent:
    """
    MedAI Knowledge Agent:
    - Builds and manages ChromaDB collections.
    - Provides structured summaries for fracture diagnoses.
    - Supports RAG over MedAI clinical/technical sources.
    """

    def __init__(self) -> None:
        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # Shared embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        # Collections
        self.diag_collection = self._setup_diag_collection()
        self.source_collection = self._setup_source_collection()

    # ----------------- Collection Setup -----------------
    def _setup_diag_collection(self):
        print("Checking/creating diagnosis collection...")
        collection = self.client.get_or_create_collection(
            name=DIAG_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

        diagnoses = list(MEDICAL_KNOWLEDGE_BASE.keys())
        ids = [d.lower().replace(" ", "-") for d in diagnoses]

        # If empty or count mismatch, repopulate
        if collection.count() != len(diagnoses):
            print(
                f"Diagnosis collection has {collection.count()} docs, "
                f"expected {len(diagnoses)}. Resetting..."
            )
            self.client.delete_collection(DIAG_COLLECTION_NAME)
            collection = self.client.get_or_create_collection(
                name=DIAG_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
            collection.add(documents=diagnoses, ids=ids)

        return collection

    def _setup_source_collection(self):
        print("Checking/creating RAG source collection...")
        collection = self.client.get_or_create_collection(
            name=SOURCE_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

        ids = [doc["id"] for doc in RAG_SOURCE_DOCS]
        docs = [
            f"Title: {doc['title']}\nCategory: {doc['category']}\n\n{doc['content']}\n\nUse case: {doc['use_case']}"
            for doc in RAG_SOURCE_DOCS
        ]
        metadatas = [
            {
                "title": doc["title"],
                "category": doc["category"],
                "use_case": doc["use_case"],
            }
            for doc in RAG_SOURCE_DOCS
        ]

        if collection.count() != len(docs):
            print(
                f"Source collection has {collection.count()} docs, "
                f"expected {len(docs)}. Resetting..."
            )
            self.client.delete_collection(SOURCE_COLLECTION_NAME)
            collection = self.client.get_or_create_collection(
                name=SOURCE_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
            collection.add(ids=ids, documents=docs, metadatas=metadatas)

        return collection

    # ----------------- Structured Summary for Diagnoses -----------------
    def get_medical_summary(self, diagnosis: str, confidence: float) -> Dict[str, Any]:
        diagnosis = diagnosis.strip()

        results = self.diag_collection.query(
            query_texts=[diagnosis],
            n_results=1,
            include=["documents", "distances"],
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return {
                "error": f"Vector search failed to find a relevant diagnosis for '{diagnosis}'."
            }

        retrieved_name = results["documents"][0][0]
        raw = MEDICAL_KNOWLEDGE_BASE.get(retrieved_name)

        if not raw:
            return {
                "error": f"Retrieved diagnosis '{retrieved_name}' not present in knowledge base."
            }

        return {
            "Diagnosis": retrieved_name,
            "Ensemble_Confidence": f"{confidence:.2f}",
            "Type_Definition": raw.get("definition"),
            "ICD_Code": raw.get("icd_code", "N/A"),
            "Severity_Rating": raw.get("severity"),
            "Treatment_Guidelines": raw.get("treatment_guidelines"),
            "Long_Term_Prognosis": raw.get("prognosis_notes"),
        }

    # ----------------- Helper for Critic Agent -----------------
    def get_context_for_label(self, label: str) -> str:
        """
        Retrieves the definition context for the Critic Agent.
        """
        # We can reuse get_medical_summary with a dummy confidence
        summary = self.get_medical_summary(label, 1.0)
        if "error" in summary:
            # Fallback based on knowledge base keys slightly matching
            # Or generic definition
            return f"Condition '{label}' regarding bone integrity."
        
        return summary.get("Type_Definition", "No definition found.")

    # ----------------- RAG over MedAI Sources -----------------
    def retrieve_sources(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        query = query.strip()
        results = self.source_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        out: List[Dict[str, Any]] = []
        for doc_text, meta in zip(docs, metas):
            out.append(
                {
                    "title": meta.get("title"),
                    "category": meta.get("category"),
                    "use_case": meta.get("use_case"),
                    "content": doc_text,
                }
            )
        return out

    # ----------------- LLaMA 3 Integration (Optional) -----------------
    def llama_available(self) -> bool:
        return bool(LLAMA_API_BASE)

    def generate_explanation_with_llama(
        self,
        summary: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        audience: str = "patient",
    ) -> Optional[str]:
        if not self.llama_available():
            return None

        system_prompt = (
            "You are the language agent in the MedAI multi-agent system. "
            "You are given:\n"
            "1) A structured fracture summary produced by a diagnostic ensemble.\n"
            "2) Retrieved domain and technical documents from MedAI's curated knowledge base.\n\n"
            "Your job is to explain the diagnosis and the system behavior using ONLY this context. "
            "Do not invent new medical facts. Do not give direct medical advice or treatment plans. "
            "Emphasize that this is informational and does not replace a clinician."
        )

        if audience == "clinician":
            user_instruction = (
                "Explain the diagnosis and relevant context to an orthopedic clinician or radiologist. "
                "Include fracture type, ICD-style coding, likely management options at a high level, "
                "and how the MedAI ensemble + Grad-CAM contribute to decision support."
            )
        else:
            user_instruction = (
                "Explain the diagnosis to a layperson patient. Use simple language to describe what "
                "the fracture means, roughly how it is treated and what recovery might involve. "
                "Avoid giving strict medical advice; encourage the patient to talk to their doctor."
            )

        docs_block = "\n\n---\n\n".join(
            f"[{d['category']}] {d['title']}\n\n{d['content']}" for d in retrieved_docs
        )

        context = (
            f"Structured summary:\n{summary}\n\n"
            f"Retrieved MedAI RAG documents:\n\n{docs_block}"
        )

        payload = {
            "model": LLAMA_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_instruction + "\n\nCONTEXT:\n" + context,
                },
            ],
            "temperature": 0.2,
        }

        try:
            with httpx.Client(base_url=LLAMA_API_BASE, timeout=60.0) as client:
                resp = client.post(
                    "/chat/completions",
                    headers={"Authorization": f"Bearer {LLAMA_API_KEY}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print("LLaMA 3 call failed:", e)
            return None


# --------------------------------------------------------------------------
# FastAPI App Setup
# --------------------------------------------------------------------------
app = FastAPI(title="MedAI Knowledge Agent API (Chroma RAG + LLaMA 3)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent Init
try:
    agent = KnowledgeAgent()
except Exception as e:
    print(f"CRITICAL ERROR during KnowledgeAgent initialization: {e}")
    agent = None
    raise e


# --------------------------------------------------------------------------
# Request Schemas
# --------------------------------------------------------------------------
class StructuredQuery(BaseModel):
    diagnosis: str
    confidence: float


class RAGSourceQuery(BaseModel):
    query: str
    top_k: int = TOP_K_RESULTS


class RAGExplanationQuery(BaseModel):
    diagnosis: str
    confidence: float
    audience: str = "patient"  # "patient" or "clinician"
    top_k_sources: int = TOP_K_RESULTS


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "MedAI Knowledge Agent is running.",
        "llama_available": agent.llama_available() if agent else False,
    }


@app.post("/retrieve_summary")
def retrieve_summary(query: StructuredQuery):
    if not agent:
        raise HTTPException(status_code=500, detail="Knowledge Agent failed to initialize.")

    summary = agent.get_medical_summary(query.diagnosis, query.confidence)
    if "error" in summary:
        return {"status": "error", "message": summary["error"]}

    return {"status": "success", "summary": summary}


@app.post("/rag_sources")
def rag_sources(query: RAGSourceQuery):
    if not agent:
        raise HTTPException(status_code=500, detail="Knowledge Agent failed to initialize.")

    docs = agent.retrieve_sources(query.query, top_k=query.top_k)
    return {"status": "success", "results": docs}


@app.post("/rag_explanation")
def rag_explanation(query: RAGExplanationQuery):
    if not agent:
        raise HTTPException(status_code=500, detail="Knowledge Agent failed to initialize.")

    summary = agent.get_medical_summary(query.diagnosis, query.confidence)
    if "error" in summary:
        return {"status": "error", "message": summary["error"]}

    docs = agent.retrieve_sources(query.diagnosis, top_k=query.top_k_sources)
    explanation = agent.generate_explanation_with_llama(summary, docs, audience=query.audience)

    return {
        "status": "success",
        "structured_summary": summary,
        "retrieved_sources": docs,
        "llama_used": explanation is not None,
        "answer": explanation or (
            "LLaMA 3 endpoint is not configured. "
            "Use 'structured_summary' and 'retrieved_sources' as context for your own LLM."
        ),
    }


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Run with: python knowledge_agent.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
