"""Enums de dominio. Sin logica, sin dependencias de infraestructura -- solo
vocabulario compartido entre schemas, servicios, agentes y persistencia."""

from enum import Enum


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    LEGAL_AGENT = "legal_agent"
    DOCUMENT_ANALYZER = "document_analyzer"
    VALIDATOR = "validator"


class LegalRank(str, Enum):
    LEY = "ley"
    DECRETO = "decreto"
    DNU = "decreto_necesidad_urgencia"
    DECRETO_LEY = "decreto_ley"
    CONSTITUCION = "constitucion"
    RESOLUCION = "resolucion"


class LegalStatus(str, Enum):
    IN_FORCE = "in_force"
    REPEALED = "repealed"
    PARTIALLY_REPEALED = "partially_repealed"
    ANNULLED = "annulled"
    EXPIRED = "expired"


class ReformQuality(str, Enum):
    CLEAN = "clean"
    PARTIAL = "partial"
    BOOTSTRAP_ONLY = "bootstrap_only"


class QueryIntent(str, Enum):
    LEGAL_CONSULTATION = "legal_consultation"
    DOCUMENT_ANALYSIS = "document_analysis"
    VERSION_DIFF = "version_diff"
    GENERAL_INQUIRY = "general_inquiry"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_FOUND = "not_found"


class ApiKeyScope(str, Enum):
    """Admin incluye todo lo que puede hacer client -- ver la jerarquia en
    app/core/security.py::require_scope."""

    CLIENT = "client"
    ADMIN = "admin"
