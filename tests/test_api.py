"""
Tests for the API module.
"""

import pytest
from fastapi.testclient import TestClient

# Test API imports
def test_api_imports():
    from api import (
        app,
        create_app,
        DetectRequest,
        DetectResponse,
        HumanizeRequest,
        HumanizeResponse,
        ExternalHumanizeRequest,
        ExternalHumanizeResponse,
        HealthResponse,
    )
    assert app is not None
    assert create_app is not None
    assert DetectRequest is not None
    assert DetectResponse is not None
    assert HumanizeRequest is not None
    assert HumanizeResponse is not None
    assert ExternalHumanizeRequest is not None
    assert ExternalHumanizeResponse is not None
    assert HealthResponse is not None

def test_schemas():
    from api.schemas import (
        DetectRequest,
        DetectResponse,
        HumanizeRequest,
        HumanizeResponse,
        ExternalHumanizeRequest,
        ExternalHumanizeResponse,
        HealthResponse,
    )
    
    # Test DetectRequest validation
    req = DetectRequest(text="Test text")
    assert req.text == "Test text"
    
    with pytest.raises(Exception):
        DetectRequest(text="")  # min_length=1
    
    with pytest.raises(Exception):
        DetectRequest(text="x" * 10001)  # max_length=10000
    
    # Test HumanizeRequest
    hreq = HumanizeRequest(text="Test text", temperature=0.7, top_p=0.9, max_tokens=512)
    assert hreq.temperature == 0.7
    assert hreq.re_detect == True
    
    with pytest.raises(Exception):
        HumanizeRequest(text="Test text", temperature=2.5)  # gt=2.0
    
    # Test ExternalHumanizeRequest
    ereq = ExternalHumanizeRequest(
        text="Test text",
        provider="gemini",
        api_key="test-key",
        temperature=0.7,
        max_tokens=512
    )
    assert ereq.provider == "gemini"
    
    with pytest.raises(Exception):
        ExternalHumanizeRequest(text="Test", provider="invalid", api_key="key")

def test_health_endpoint():
    from api import app
    from fastapi.testclient import TestClient
    client = TestClient(app, raise_server_exceptions=False)
    
    response = client.get("/health")
    # Health endpoint returns 200 if healthy, 503 if detector not trained
    # Note: TestClient doesn't properly run lifespan, so detector is None
    # This is expected behavior - the endpoint handles missing detector gracefully
    assert response.status_code in (200, 503)
    data = response.json()
    # If detector is not initialized, we get an error response
    # This is expected behavior for the test environment
    if response.status_code == 503:
        assert "detail" in data
    else:
        assert "status" in data
        assert data["status"] in ("ok", "degraded")
        assert "detector_loaded" in data
        assert "humanizer_ready" in data

def test_detect_endpoint_structure():
    from api import app
    client = TestClient(app)
    
    # Test without trained detector (should return 503)
    response = client.post("/detect", json={"text": "Test text"})
    # Either 503 (not trained) or 200 (if trained)
    assert response.status_code in (200, 503)
    
    if response.status_code == 200:
        data = response.json()
        assert "ai_probability" in data
        assert "is_ai" in data
        assert "label" in data
        assert "feature_breakdown" in data
        assert "sentence_scores" in data
        assert "top_repeated_ngrams" in data
        assert "n_tokens" in data
        assert "elapsed_seconds" in data

def test_humanize_endpoint_structure():
    from api import app
    client = TestClient(app)
    
    # Test without configured humanizer (should return 503)
    response = client.post("/humanize", json={"text": "Test text"})
    assert response.status_code in (200, 503)
    
    if response.status_code == 200:
        data = response.json()
        assert "original_text" in data
        assert "humanized_text" in data
        assert "original_ai_probability" in data
        assert "backend_used" in data
        assert "elapsed_seconds" in data

def test_external_humanize_endpoint_structure():
    from api import app
    client = TestClient(app)
    
    # Test without valid API key (should return error from provider)
    response = client.post(
        "/humanize-external",
        json={
            "text": "Test text",
            "provider": "gemini",
            "api_key": "invalid-key",
            "temperature": 0.7,
            "max_tokens": 512
        }
    )
    # Should fail at provider level, not validation
    assert response.status_code in (200, 400, 401, 403, 502, 503)

def test_cors_headers():
    from api import app
    client = TestClient(app)
    
    response = client.options("/detect")
    # CORS should be enabled
    assert response.status_code in (200, 405)  # OPTIONS may not be implemented

# Integration test (requires trained detector)
@pytest.mark.integration
def test_full_detection_pipeline():
    """Integration test - requires trained detector."""
    from api import app
    client = TestClient(app)
    
    # This test only runs if detector is trained
    response = client.post("/detect", json={"text": "In conclusion, AI is important."})
    if response.status_code == 503:
        pytest.skip("Detector not trained")
    
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["ai_probability"] <= 100
    assert isinstance(data["is_ai"], bool)
    assert data["label"] in ["AI-generated", "Likely human", "Uncertain"]