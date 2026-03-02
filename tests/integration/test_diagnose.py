import io
import os
import sys
import json
from fastapi.testclient import TestClient
# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend_hf.app import app
from PIL import Image

client = TestClient(app)

def test_diagnose_report_and_audit_log():
    # create a small RGB image
    img = Image.new('RGB', (224,224), color=(128,128,128))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    files = {'file': ('test.png', buf, 'image/png')}
    data = {'format': 'json'}

    resp = client.post('/diagnose/report', files=files, data=data)
    assert resp.status_code == 200
    j = resp.json()
    # basic structure
    assert 'prediction' in j
    assert 'explanation' in j

    # audit log should be created
    audit = j.get('audit')
    assert audit and 'inference_id' in audit
    inference_id = audit['inference_id']
    logs_dir = os.path.join('outputs', 'inference_logs')
    log_path = os.path.join(logs_dir, f"{inference_id}.json")
    assert os.path.exists(log_path)
    with open(log_path, 'r') as fh:
        rec = json.load(fh)
        assert rec.get('inference_id') == inference_id
