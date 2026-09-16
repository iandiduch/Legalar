"""Scanner heuristico de patrones de prompt injection sobre texto de documentos
recien parseados, antes de indexarlos (ver app/services/ingestion_worker.py).

Es una capa de defensa en profundidad, no una garantia: un heuristico basado en
regex nunca cubre todas las variantes posibles de injection (paraphrasis,
ofuscacion, otros idiomas no listados aca). Las mitigaciones que sostienen la
seguridad real son las otras dos: el hardening explicito en los prompts de
agentes (app/agents/prompts/defaults/*.md) y el delimitado del contexto RAG
como dato no confiable (app/agents/specialized/knowledge_agent.py). Este
modulo solo agrega una red adicional para los intentos mas obvios/comunes,
y deja rastro auditable (job FAILED con el detalle de que patron matcheo)
en vez de indexar en silencio.
"""

import re

_QUALIFIER_EN = r"(?:all|any|previous|prior|above)"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions_en",
        re.compile(rf"ignore\s+{_QUALIFIER_EN}(?:\s+{_QUALIFIER_EN})*\s+instructions", re.IGNORECASE),
    ),
    (
        "ignore_instructions_es",
        re.compile(r"ignor[aá]\s+(todas\s+)?las\s+instrucciones\s+(anteriores|previas)", re.IGNORECASE),
    ),
    ("disregard_above_en", re.compile(r"disregard\s+(the\s+)?(above|previous|prior)", re.IGNORECASE)),
    ("fake_role_marker", re.compile(r"\b(system|assistant|user)\s*:", re.IGNORECASE)),
    ("new_instructions_en", re.compile(r"new\s+instructions\s*:", re.IGNORECASE)),
    ("new_instructions_es", re.compile(r"nuevas\s+instrucciones\s*:", re.IGNORECASE)),
    ("reveal_prompt_en", re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.IGNORECASE)),
    ("reveal_prompt_es", re.compile(r"revel[aá]\s+(tu|el)\s+(prompt|instrucciones\s+del\s+sistema)", re.IGNORECASE)),
    ("role_override_en", re.compile(r"you\s+are\s+now\s+(a|an)\b", re.IGNORECASE)),
    ("role_override_es", re.compile(r"actu[aá]\s+como\s+(si\s+fueras\s+)?(un|una)\b", re.IGNORECASE)),
    ("jailbreak_marker", re.compile(r"\b(jailbreak|do\s+anything\s+now|dan\s+mode)\b", re.IGNORECASE)),
    (
        "override_rules_en",
        re.compile(r"override\s+(your|the|previous)\s+(previous\s+)?(rules|instructions|guidelines)", re.IGNORECASE),
    ),
]


def scan_for_injection_patterns(text: str) -> list[str]:
    """Devuelve los nombres de los patrones que matchearon en `text` (lista
    vacia si no hay ninguno). No lanza excepcion: el caller decide que hacer
    con el resultado (ver ingestion_worker._process_job)."""
    return [name for name, pattern in _PATTERNS if pattern.search(text)]
