"""Servicio seguro de lectura, extracción y sanitización de enlaces web (Anti-SSRF & Anti-Injection)."""

import html
import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

from app.core.prompt_injection_scanner import scan_for_injection_patterns

logger = logging.getLogger(__name__)

# Expresión regular robusta para detectar URLs HTTP/HTTPS en el texto del usuario
_RE_URL = re.compile(
    r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*",
    re.IGNORECASE,
)

# Nombres de host locales o de infraestructura que deben bloquearse sin resolución
BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "host.docker.internal",
    "postgres",
    "redis",
    "db",
    "minio",
}

# Etiquetas HTML que se descartan completamente por contener código, publicidad o navegación
IGNORED_HTML_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "button",
    "iframe",
}


class WebPageResult(BaseModel):
    """Resultado estructurado de la lectura y extracción de un enlace web."""

    success: bool
    url: str
    domain: str
    title: str | None = None
    clean_text: str = ""
    error: str | None = None


def find_urls_in_text(text: str) -> list[str]:
    """Extrae todas las URLs válidas presentes en una cadena de texto."""
    if not text:
        return []
    matches = _RE_URL.findall(text)
    # Deduplicar manteniendo orden
    seen = set()
    urls = []
    for m in matches:
        clean = m.rstrip(".,;:)\"'>")
        if clean not in seen:
            seen.add(clean)
            urls.append(clean)
    return urls


def is_safe_public_url(url: str) -> tuple[bool, str]:
    """Audita una URL contra ataques SSRF (Server-Side Request Forgery).

    Valida que el protocolo sea HTTP/HTTPS y que la IP de destino sea pública,
    bloqueando direcciones privadas (RFC 1918), loopback, link-local y redes internas.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "Esquema no permitido. Solo se admiten enlaces 'http' o 'https'."

        hostname = parsed.hostname
        if not hostname:
            return False, "La URL no posee un nombre de host o dominio válido."

        hostname_lower = hostname.lower()
        if hostname_lower in BLOCKED_HOSTNAMES or hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
            return False, "Acceso denegado: El destino apunta a una red local o de infraestructura interna."

        # Resolución DNS y validación de cada IP devuelta
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            return False, f"No se pudo resolver el dominio del enlace ({exc})."

        for entry in addr_info:
            ip_str = entry[4][0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                if (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_reserved
                    or ip_obj.is_multicast
                    or ip_obj.is_unspecified
                ):
                    return False, f"Acceso denegado por seguridad (Anti-SSRF): La IP de destino ({ip_str}) es privada o restringida."
            except ValueError:
                return False, f"Dirección IP no válida: {ip_str}"

        return True, ""
    except Exception as exc:
        return False, f"Fallo al validar URL: {exc}"


class HTMLTextExtractor(HTMLParser):
    """Parser liviano y de alto rendimiento basado en html.parser estándar para extraer texto limpio."""

    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in IGNORED_HTML_TAGS:
            self.skip_depth += 1
        elif tag_lower == "title":
            self.in_title = True
        elif tag_lower in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section"):
            if self.skip_depth == 0 and self.text_parts and not self.text_parts[-1].endswith("\n"):
                self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in IGNORED_HTML_TAGS:
            if self.skip_depth > 0:
                self.skip_depth -= 1
        elif tag_lower == "title":
            self.in_title = False
        elif tag_lower in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section"):
            if self.skip_depth == 0 and self.text_parts and not self.text_parts[-1].endswith("\n"):
                self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if not stripped:
            return

        if self.in_title:
            self.title_parts.append(stripped)
        elif self.skip_depth == 0:
            self.text_parts.append(data)

    def get_clean_text(self) -> tuple[str, str]:
        raw_title = " ".join(self.title_parts).strip()
        clean_title = html.unescape(raw_title)

        raw_body = "".join(self.text_parts)
        # Normalizar saltos de línea repetidos
        clean_body = re.sub(r"\n{3,}", "\n\n", raw_body).strip()
        # Normalizar espacios dentro de líneas
        clean_body = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in clean_body.splitlines())
        clean_body = html.unescape(clean_body)
        return clean_title, clean_body


async def fetch_web_page_content(
    url: str,
    timeout_seconds: float = 5.0,
    max_chars: int = 15000,
) -> WebPageResult:
    """Descarga de forma segura y extrae el texto principal de una página web.

    Aplica defensas Anti-SSRF, límites estrictos de latencia/tamaño y auditoría
    contra ataques de Indirect Prompt Injection.
    """
    domain = urlparse(url).netloc or "enlace-externo"

    # 1. Auditoría Anti-SSRF antes de disparar la conexión
    is_safe, safety_err = is_safe_public_url(url)
    if not is_safe:
        logger.warning("Bloqueo Anti-SSRF para URL '%s': %s", url, safety_err)
        return WebPageResult(
            success=False,
            url=url,
            domain=domain,
            error=f"No es posible acceder al enlace por motivos de seguridad: {safety_err}",
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "LegalarBot/1.0 (Legal Research Assistant; +https://legalar.onys.app)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    }

    current_url = url
    max_redirects = 3
    redirect_count = 0

    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=False,
        ) as client:
            while True:
                resp = await client.get(current_url, headers=headers)

                if resp.is_redirect:
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        return WebPageResult(
                            success=False,
                            url=url,
                            domain=domain,
                            error="Demasiadas redirecciones consecutivas (máximo 3).",
                        )
                    location = resp.headers.get("location")
                    if not location:
                        break
                    next_url = urljoin(current_url, location)
                    is_safe_redirect, redirect_err = is_safe_public_url(next_url)
                    if not is_safe_redirect:
                        logger.warning("Bloqueo Anti-SSRF en redirección hacia '%s': %s", next_url, redirect_err)
                        return WebPageResult(
                            success=False,
                            url=url,
                            domain=domain,
                            error=f"Redirección bloqueada por seguridad (Anti-SSRF): {redirect_err}",
                        )
                    current_url = next_url
                    continue
                break

            if resp.status_code >= 400:
                logger.info("El sitio web respondió con código HTTP %d para URL: %s", resp.status_code, current_url)
                return WebPageResult(
                    success=False,
                    url=url,
                    domain=domain,
                    error=f"El servidor del enlace respondió con código HTTP {resp.status_code}.",
                )

            content_type = resp.headers.get("content-type", "").lower()
            # Si el contenido es binario (PDF/Imagen), extraer aviso para indicar que use la subida de archivos
            if "pdf" in content_type:
                return WebPageResult(
                    success=False,
                    url=url,
                    domain=domain,
                    error="El enlace apunta a un archivo PDF binario. Podés descargarlo y adjuntarlo mediante el clip en el chat para auditarlo de forma íntegra.",
                )

            # Extraer texto limpio del HTML
            raw_html = resp.text
            # Limitar tamaño de HTML procesado a 2 MB
            if len(raw_html) > 2 * 1024 * 1024:
                raw_html = raw_html[: 2 * 1024 * 1024]

            extractor = HTMLTextExtractor()
            extractor.feed(raw_html)
            title, body = extractor.get_clean_text()

            if not body or len(body.strip()) < 40:
                return WebPageResult(
                    success=False,
                    url=url,
                    domain=domain,
                    error="No se pudo extraer contenido textual legible de la página web (posible sitio dinámico o protegido por captcha/paywall).",
                )

            # 2. Auditoría activa contra Indirect Prompt Injection
            matches = scan_for_injection_patterns(body)
            if matches:
                matched_str = ", ".join(matches)
                logger.warning("Indirect Prompt Injection detectado en URL '%s': %s", url, matched_str)
                return WebPageResult(
                    success=False,
                    url=url,
                    domain=domain,
                    error=(
                        f"El contenido de la página web fue bloqueado por seguridad: contiene patrones "
                        f"de instrucciones sospechosas dirigidas a manipular al asistente ({matched_str})."
                    ),
                )

            # Truncar si excede el límite de caracteres para no saturar el contexto
            if len(body) > max_chars:
                body = body[:max_chars] + f"\n\n[... Contenido truncado a {max_chars} caracteres para análisis jurídico ...]"

            return WebPageResult(
                success=True,
                url=url,
                domain=domain,
                title=title or None,
                clean_text=body,
            )

    except httpx.TimeoutException:
        logger.info("Timeout de %s segundos al consultar URL: %s", timeout_seconds, url)
        return WebPageResult(
            success=False,
            url=url,
            domain=domain,
            error=f"Tiempo de espera agotado: El servidor de {domain} tardó más de {int(timeout_seconds)}s en responder.",
        )
    except Exception as exc:
        logger.warning("Fallo al descargar URL '%s': %s", url, exc)
        return WebPageResult(
            success=False,
            url=url,
            domain=domain,
            error=f"No fue posible conectar con el sitio web ({exc}).",
        )
