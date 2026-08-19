"""Tests for get_patient_context MCP tool."""
import json


from rag_document_intelligence.mcp.tools import handle_get_patient_context
from rag_document_intelligence.patient_context import (
    PatientContext,
    PatientNotFoundError,
)


def test_get_patient_context_valid(mock_patient_context, mock_patient_provider):
    provider_factory = lambda: mock_patient_provider

    result = handle_get_patient_context(
        {"patient_id": "P001"},
        patient_provider_factory=provider_factory,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["patient_id"] == "P001"
    assert data["medical_conditions"] == ["hypertension"]
    assert data["contraindications"] == ["avoid heavy resistance"]
    assert "demographics" in data
    assert data["demographics"]["age"] == 45


def test_get_patient_context_missing_patient():
    class _MissingProvider:
        def get_patient_context(self, patient_id):
            raise PatientNotFoundError(f"Patient '{patient_id}' not found")
        def patient_exists(self, patient_id):
            return False

    result = handle_get_patient_context(
        {"patient_id": "MISSING"},
        patient_provider_factory=lambda: _MissingProvider(),
    )
    assert result.is_error is True
    assert "MISSING" in result.content[0].text or "not found" in result.content[0].text.lower()


def test_get_patient_context_empty_id():
    result = handle_get_patient_context(
        {"patient_id": ""},
    )
    assert result.is_error is True
    assert "patient id" in result.content[0].text.lower() or "required" in result.content[0].text.lower()


def test_get_patient_context_none_id():
    result = handle_get_patient_context(
        {"patient_id": None},
    )
    assert result.is_error is True


def test_get_patient_context_path_traversal_blocked():
    result = handle_get_patient_context(
        {"patient_id": "../../../etc/passwd"},
    )
    assert result.is_error is True
    assert "invalid characters" in result.content[0].text.lower()


def test_get_patient_context_oversized_id():
    result = handle_get_patient_context(
        {"patient_id": "x" * 200},
    )
    assert result.is_error is True
    assert "too long" in result.content[0].text.lower()


def test_get_patient_context_no_filesystem_path_exposed(mock_patient_context, mock_patient_provider):
    """Ensure no internal filesystem paths are exposed."""
    result = handle_get_patient_context(
        {"patient_id": "P001"},
        patient_provider_factory=lambda: mock_patient_provider,
    )

    data = json.loads(result.content[0].text)
    text = json.dumps(data)
    assert "data/patients" not in text
    assert ".json" not in text
    assert "/tmp" not in text


def test_get_patient_context_provider_init_failure():
    result = handle_get_patient_context(
        {"patient_id": "P001"},
        patient_provider_factory=lambda: (_ for _ in ()).throw(RuntimeError("Config error")),
    )
    assert result.is_error is True
    assert "Failed to initialize" in result.content[0].text


def test_get_patient_context_load_failure():
    class _FailingProvider:
        def get_patient_context(self, patient_id):
            raise RuntimeError("Backend API unreachable")
        def patient_exists(self, patient_id):
            return True

    result = handle_get_patient_context(
        {"patient_id": "P001"},
        patient_provider_factory=lambda: _FailingProvider(),
    )
    assert result.is_error is True


def test_get_patient_context_no_secrets(mock_patient_context, mock_patient_provider):
    """Ensure no secret/health identifiers are over-exposed."""
    # Add sensitive-looking fields to patient context
    patient = PatientContext(
        patient_id="P001",
        doctor_notes="Patient has controlled hypertension.",
    )

    class _Provider:
        def get_patient_context(self, patient_id):
            return patient
        def patient_exists(self, patient_id):
            return True

    result = handle_get_patient_context(
        {"patient_id": "P001"},
        patient_provider_factory=lambda: _Provider(),
    )

    data = json.loads(result.content[0].text)
    # doctor_notes should be truncated, not absent
    assert "doctor_notes" in data
    # No raw filesystem paths
    text = json.dumps(data)
    assert "PATIENT_DATA_DIR" not in text
    assert "os.environ" not in text


def test_get_patient_context_patient_id_requested_correctly(
    mock_patient_context, mock_patient_provider
):
    result = handle_get_patient_context(
        {"patient_id": "P001"},
        patient_provider_factory=lambda: mock_patient_provider,
    )

    assert result.is_error is False
    assert mock_patient_provider.requested_id == "P001"
