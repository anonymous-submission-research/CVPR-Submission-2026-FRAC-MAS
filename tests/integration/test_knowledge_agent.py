import httpx
import json
import os

def test_knowledge_agent():
    print("\n" + "="*60)
    print("      MEDAI 2026: KNOWLEDGE AGENT MULTIMODAL TEST")
    print("="*60)

    url = "http://localhost:8000/rag_explanation"
    
    # Real sample from the dataset
    real_image_path = "balanced_augmented_dataset/test/Spiral/Spiral_141_jpg.rf.f9720705840a6ef6dda2dc1341026d64_0000.jpg"
    
    payload = {
        "diagnosis": "Spiral",
        "confidence": 0.95,
        "audience": "patient",
        "image_path": real_image_path
    }

    print(f"\n[*] Sending Request to Knowledge Agent...")
    print(f"[*] Payload: {json.dumps(payload, indent=2)}")

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            print("\n" + "-"*60)
            print("KNOWLEDGE AGENT RESPONSE")
            print("-" * 60)
            
            if data["status"] == "success":
                print(f"\n[STRUCTURED SUMMARY]:")
                print(json.dumps(data["structured_summary"], indent=2))
                
                print(f"\n[RETRIEVED SOURCES]:")
                for src in data["retrieved_sources"]:
                    print(f"- {src['title']} ({src['category']})")
                
                print(f"\n[MEDGEMMA VLM EXPLANATION]:")
                print(data["answer"])
                
                print(f"\n[METADATA]:")
                print(f"Multimodal used: {data['llama_used']}") # Keeping key name for compatibility
            else:
                print(f"Error: {data['message']}")
                
    except Exception as e:
        print(f"\n[!] Connection Failed: {e}")
        print("Note: Ensure the knowledge agent server and your MedGemma endpoint are running.")

    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    test_knowledge_agent()
