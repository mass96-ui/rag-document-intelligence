"""Tests for patient context models and providers."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_document_intelligence.patient_context import (
    Demographics,
    LocalPatientContextProvider,
    MLSessionResult,
    NullPatientContextProvider,
    PatientContext,
    PatientContextError,
    PatientNotFoundError,
    get_patient_context_provider,
)


# ---------------------------------------------------------------------------
# Demographics tests
# ---------------------------------------------------------------------------

def test_demographics_all_fields():
    demo = Demographics(
        height_cm=175.5,
        weight_kg=82.0,
        age=45,
        sex="male",
        bmi=26.7,
    )
    assert demo.height_cm == 175.5
    assert demo.weight_kg == 82.0
    assert demo.age == 45
    assert demo.sex == "male"
    assert demo.bmi == 26.7


def test_demographics_optional_fields_default_none():
    demo = Demographics()
    assert demo.height_cm is None
    assert demo.weight_kg is None
    assert demo.age is None
    assert demo.sex is None
    assert demo.bmi is None


def test_demographics_rejects_non_numeric_height():
    with pytest.raises(ValidationError):
        Demographics(height_cm="not a number")


def test_demographics_negative_weight_accepted():
    """Pydantic allows negative floats for Optional[float] fields.
    Validation of medical ranges is the application's responsibility."""
    demo = Demographics(weight_kg=-5.0)
    assert demo.weight_kg == -5.0


# ---------------------------------------------------------------------------
# PatientContext tests
# ---------------------------------------------------------------------------

def test_patient_context_valid():
    patient = PatientContext(
        patient_id="P001",
        demographics=Demographics(age=45, sex="male"),
        medical_conditions=["hypertension"],
        current_medications=["lisinopril"],
    )
    assert patient.patient_id == "P001"
    assert patient.demographics.age == 45
    assert patient.medical_conditions == ["hypertension"]
    assert patient.current_medications == ["lisinopril"]


def test_patient_context_requires_patient_id():
    with pytest.raises(ValidationError):
        PatientContext()


def test_patient_context_optional_fields_default_none():
    patient = PatientContext(patient_id="P002")
    assert patient.demographics is None
    assert patient.medical_conditions is None
    assert patient.diabetes_status is None
    assert patient.current_medications is None
    assert patient.contraindications is None


def test_patient_context_allows_extra_fields():
    patient = PatientContext(
        patient_id="P003",
        custom_field="extra_value",
    )
    assert patient.patient_id == "P003"


def test_patient_context_list_fields_can_be_empty():
    patient = PatientContext(
        patient_id="P004",
        medical_conditions=[],
        current_medications=[],
    )
    assert patient.medical_conditions == []
    assert patient.current_medications == []


# ---------------------------------------------------------------------------
# MLSessionResult tests
# ---------------------------------------------------------------------------

def test_ml_session_result_all_fields():
    ml = MLSessionResult(
        activation_score=0.87,
        repetition_count=12,
        resistance=35.0,
        movement_quality=0.91,
        fatigue_score=0.3,
        timestamp="2026-08-19T11:00:00Z",
        model_version="v1.2.3",
    )
    assert ml.activation_score == 0.87
    assert ml.repetition_count == 12
    assert ml.resistance == 35.0
    assert ml.movement_quality == 0.91
    assert ml.fatigue_score == 0.3
    assert ml.timestamp == "2026-08-19T11:00:00Z"
    assert ml.model_version == "v1.2.3"


def test_ml_session_result_all_optional():
    ml = MLSessionResult()
    assert ml.activation_score is None
    assert ml.repetition_count is None
    assert ml.resistance is None
    assert ml.movement_quality is None
    assert ml.fatigue_score is None
    assert ml.timestamp is None
    assert ml.model_version is None


def test_ml_session_result_allows_extra_fields():
    ml = MLSessionResult(extra_metric=42)
    assert ml.extra_metric == 42


# ---------------------------------------------------------------------------
# LocalPatientContextProvider tests
# ---------------------------------------------------------------------------

def test_local_provider_loads_existing_patient(tmp_path):
    patients_dir = tmp_path / "patients"
    patients_dir.mkdir()
    patient_file = patients_dir / "P001.json"
    patient_file.write_text(json.dumps({
        "patient_id": "P001",
        "medical_conditions": ["hypertension"],
        "demographics": {"age": 45},
    }), encoding="utf-8")

    provider = LocalPatientContextProvider(data_dir=patients_dir)
    context = provider.get_patient_context("P001")

    assert context.patient_id == "P001"
    assert context.medical_conditions == ["hypertension"]
    assert context.demographics.age == 45


def test_local_provider_patient_exists(tmp_path):
    patients_dir = tmp_path / "patients"
    patients_dir.mkdir()
    (patients_dir / "P001.json").write_text(
        json.dumps({"patient_id": "P001"}), encoding="utf-8"
    )

    provider = LocalPatientContextProvider(data_dir=patients_dir)
    assert provider.patient_exists("P001") is True
    assert provider.patient_exists("P999") is False


def test_local_provider_missing_patient_raises(tmp_path):
    patients_dir = tmp_path / "patients"
    patients_dir.mkdir()

    provider = LocalPatientContextProvider(data_dir=patients_dir)
    with pytest.raises(PatientNotFoundError):
        provider.get_patient_context("MISSING")


def test_local_provider_directory_missing_raises(tmp_path):
    provider = LocalPatientContextProvider(
        data_dir=tmp_path / "nonexistent"
    )
    with pytest.raises(PatientNotFoundError, match="does not exist"):
        provider.get_patient_context("P001")


def test_local_provider_malformed_json_raises(tmp_path):
    patients_dir = tmp_path / "patients"
    patients_dir.mkdir()
    (patients_dir / "P001.json").write_text(
        "{invalid json content", encoding="utf-8"
    )

    provider = LocalPatientContextProvider(data_dir=patients_dir)
    with pytest.raises(PatientContextError, match="Malformed JSON"):
        provider.get_patient_context("P001")


def test_local_provider_invalid_data_raises(tmp_path):
    patients_dir = tmp_path / "patients"
    patients_dir.mkdir()
    (patients_dir / "P001.json").write_text(
        json.dumps({"patient_id": 12345}), encoding="utf-8"
    )

    provider = LocalPatientContextProvider(data_dir=patients_dir)
    with pytest.raises(PatientContextError, match="Failed to parse"):
        provider.get_patient_context("P001")


def test_local_provider_validates_patient_id_safety():
    provider = LocalPatientContextProvider()
    with pytest.raises(PatientNotFoundError):
        provider.get_patient_context("../../../etc/passwd")


# ---------------------------------------------------------------------------
# NullPatientContextProvider tests
# ---------------------------------------------------------------------------

def test_null_provider_always_not_found():
    provider = NullPatientContextProvider()
    assert provider.patient_exists("any_id") is False
    with pytest.raises(PatientNotFoundError):
        provider.get_patient_context("any_id")


# ---------------------------------------------------------------------------
# Provider factory tests
# ---------------------------------------------------------------------------

def test_get_patient_context_provider_local(tmp_path, monkeypatch):
    monkeypatch.setenv("PATIENT_CONTEXT_PROVIDER", "local")
    from rag_document_intelligence import config
    monkeypatch.setattr(config, "PATIENT_CONTEXT_PROVIDER", "local")
    provider = get_patient_context_provider("local")
    assert isinstance(provider, LocalPatientContextProvider)


def test_get_patient_context_provider_none():
    provider = get_patient_context_provider("none")
    assert isinstance(provider, NullPatientContextProvider)


def test_get_patient_context_provider_default(monkeypatch):
    provider = get_patient_context_provider(None)
    assert isinstance(provider, LocalPatientContextProvider)


def test_get_patient_context_provider_invalid():
    with pytest.raises(ValueError, match="Unsupported"):
        get_patient_context_provider("unknown_provider")


# ---------------------------------------------------------------------------
# Fixture file tests
# ---------------------------------------------------------------------------

def test_synthetic_fixture_p001():
    """Verify the P001 fixture loads and has expected structure."""
    provider = LocalPatientContextProvider()
    if not provider.patient_exists("P001"):
        pytest.skip("P001 fixture not present in default data directory")

    context = provider.get_patient_context("P001")
    assert context.patient_id == "P001"
    assert context.demographics is not None
    assert context.demographics.age is not None
    assert isinstance(context.medical_conditions, list)


def test_synthetic_fixture_p004_minimal():
    """Verify P004 loads with minimal fields."""
    provider = LocalPatientContextProvider()
    if not provider.patient_exists("P004"):
        pytest.skip("P004 fixture not present in default data directory")

    context = provider.get_patient_context("P004")
    assert context.patient_id == "P004"
    assert context.medical_conditions == []
    assert context.current_medications == []
    assert context.contraindications == []
