import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import PATIENT_DATA_DIR

logger = logging.getLogger(__name__)


class Demographics(BaseModel):
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    bmi: Optional[float] = None


class PatientContext(BaseModel):
    """Typed model for patient-specific context.

    All fields are optional. Absence of a field indicates the
    data is unavailable — it does NOT imply a medical condition
    or a normal value.
    """

    patient_id: str
    demographics: Optional[Demographics] = None
    medical_conditions: Optional[List[str]] = None
    diabetes_status: Optional[str] = None
    hemoglobin: Optional[float] = None
    hemoglobin_a1c: Optional[float] = None
    relevant_medical_history: Optional[str] = None
    doctor_notes: Optional[str] = None
    current_medications: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None
    rehabilitation_stage: Optional[str] = None
    exercise_history: Optional[str] = None
    session_history: Optional[List[Dict[str, Any]]] = None
    latest_ml_results: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class MLSessionResult(BaseModel):
    """Typed model for ML subsystem outputs.

    Every field is optional. The RAG does NOT calculate these;
    they are measurements produced by the ML subsystem.
    """

    activation_score: Optional[float] = None
    repetition_count: Optional[int] = None
    resistance: Optional[float] = None
    movement_quality: Optional[float] = None
    fatigue_score: Optional[float] = None
    timestamp: Optional[str] = None
    model_version: Optional[str] = None

    model_config = {"extra": "allow"}


class PatientNotFoundError(Exception):
    """Raised when a patient ID is not found in the data store."""


class PatientContextError(Exception):
    """Raised when patient data is malformed or cannot be loaded."""


class PatientContextProvider(ABC):
    """Abstract base for patient context providers.

    The RAG core does not care where patient context comes from.
    LocalPatientContextProvider loads from JSON fixtures for
    development. A future BackendPatientContextProvider will
    fetch from the SoleusAI backend database.
    """

    @abstractmethod
    def get_patient_context(
        self, patient_id: str
    ) -> PatientContext:
        """Return patient context for *patient_id*.

        Raises PatientNotFoundError if the patient does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    def patient_exists(self, patient_id: str) -> bool:
        """Return True if *patient_id* is known to this provider."""
        raise NotImplementedError


class LocalPatientContextProvider(PatientContextProvider):
    """Load synthetic development patient data from local JSON files.

    Data is stored as ``data/patients/<patient_id>.json``.
    All fixtures are synthetic — NOT FOR CLINICAL USE.
    """

    def __init__(
        self,
        data_dir: Path = PATIENT_DATA_DIR,
    ):
        self.data_dir = Path(data_dir)

    def _patient_file_path(self, patient_id: str) -> Path:
        return self.data_dir / f"{patient_id}.json"

    def patient_exists(self, patient_id: str) -> bool:
        return self._patient_file_path(patient_id).exists()

    def get_patient_context(
        self, patient_id: str
    ) -> PatientContext:
        if not self.data_dir.exists():
            raise PatientNotFoundError(
                f"Patient data directory does not exist: {self.data_dir}"
            )

        file_path = self._patient_file_path(patient_id)

        if not file_path.exists():
            raise PatientNotFoundError(
                f"Patient '{patient_id}' not found in {self.data_dir}"
            )

        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PatientContextError(
                f"Malformed JSON in patient file {file_path}: {exc}"
            ) from exc

        try:
            return PatientContext(**data)
        except Exception as exc:
            raise PatientContextError(
                f"Failed to parse patient data for '{patient_id}': {exc}"
            ) from exc


class NullPatientContextProvider(PatientContextProvider):
    """A provider that never returns patient context.

    Used when no patient is needed (pure knowledge-mode RAG)
    or when the backend provider is not yet configured.
    """

    def get_patient_context(
        self, patient_id: str
    ) -> PatientContext:
        raise PatientNotFoundError(
            f"No patient context provider is configured for "
            f"patient '{patient_id}'."
        )

    def patient_exists(self, patient_id: str) -> bool:
        return False


class BackendPatientContextProvider(PatientContextProvider):
    """Adapter for the future SoleusAI backend patient context API.

    This provider is an interface definition, not a hard-coded dependency.
    It defines the expected contract for fetching patient context from
    the SoleusAI backend.

    To use:
        1. Subclass this provider for a specific backend implementation.
        2. Implement ``_fetch_patient_context`` to call the backend API.
        3. Set ``PATIENT_CONTEXT_PROVIDER=backend`` in ``.env``.

    The backend API contract (expected by ``_fetch_patient_context``):
        Input:  patient_id (str)
        Output: dict with fields matching ``PatientContext`` model
                (or a JSON-serializable dict that can be parsed into
                a ``PatientContext`` instance)

    Error handling:
        - ``PatientNotFoundError``: backend returns 404 or patient not found
        - ``PatientContextError``: backend returns malformed data or other
          errors

    This implementation raises ``NotImplementedError`` in the abstract
    ``_fetch_patient_context`` method. A real backend implementation should
    subclass this and provide the actual API call logic.
    """

    def get_patient_context(
        self, patient_id: str
    ) -> PatientContext:
        if not patient_id or not patient_id.strip():
            raise PatientNotFoundError(
                "Patient ID is required."
            )

        try:
            raw_data = self._fetch_patient_context(patient_id)
        except PatientNotFoundError:
            raise
        except Exception as exc:
            logger.warning(
                "Backend patient context fetch failed for '%s': %s",
                patient_id, exc,
            )
            raise PatientContextError(
                f"Failed to fetch patient context for '{patient_id}': {exc}"
            ) from exc

        if raw_data is None:
            raise PatientNotFoundError(
                f"Patient '{patient_id}' was not found by the backend."
            )

        try:
            return PatientContext(**raw_data)
        except Exception as exc:
            raise PatientContextError(
                f"Failed to parse backend patient data for "
                f"'{patient_id}': {exc}"
            ) from exc

    def patient_exists(self, patient_id: str) -> bool:
        try:
            self.get_patient_context(patient_id)
            return True
        except PatientNotFoundError:
            return False
        except PatientContextError:
            return False

    def _fetch_patient_context(
        self, patient_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch raw patient data from the backend.

        Subclasses must implement this method to call the actual
        SoleusAI backend API and return a dict that can be parsed
        into a ``PatientContext`` instance.

        Return ``None`` if the patient is not found.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError(
            "BackendPatientContextProvider._fetch_patient_context "
            "must be implemented by a subclass."
        )


def get_patient_context_provider(
    provider_name: Optional[str] = None,
) -> PatientContextProvider:
    """Create a patient context provider from configuration."""

    from .config import PATIENT_CONTEXT_PROVIDER

    provider = (
        provider_name or PATIENT_CONTEXT_PROVIDER
    ).lower().strip()

    if provider == "local":
        return LocalPatientContextProvider()

    if provider == "none":
        return NullPatientContextProvider()

    if provider == "backend":
        return BackendPatientContextProvider()

    raise ValueError(
        f"Unsupported patient context provider: '{provider}'. "
        f"Supported: 'local', 'none', 'backend'."
    )
