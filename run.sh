#!/usr/bin/env bash
# Levanta todo el stack con un solo comando en Linux / Ubuntu.
# Uso: ./run.sh [--with-phoenix]
set -euo pipefail
cd "$(dirname "$0")"

# 1. Asegurar permisos para uploads en Linux (evita problemas de permisos con appuser uid 1000)
mkdir -p data/uploads
chmod 777 data/uploads 2>/dev/null || true

# 2. Verificar o instalar Docker automáticamente si no está instalado
if ! command -v docker &> /dev/null; then
    echo "================================================================" >&2
    echo " Docker no está instalado en este sistema." >&2
    echo " Instalando Docker Engine y Docker Compose automáticamente..." >&2
    echo "================================================================" >&2
    if command -v curl &> /dev/null; then
        curl -fsSL https://get.docker.com | sh
    elif command -v wget &> /dev/null; then
        wget -qO- https://get.docker.com | sh
    else
        echo "Error: Se requiere 'curl' o 'wget' para instalar Docker automáticamente." >&2
        echo "Por favor instalalo con: apt-get update && apt-get install -y curl" >&2
        exit 1
    fi
    systemctl enable --now docker 2>/dev/null || service docker start 2>/dev/null || true
fi

# Iniciar servicio Docker si no está corriendo
if ! docker info &> /dev/null; then
    echo "Iniciando servicio de Docker..." >&2
    systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true
fi

# 3. Detectar comando docker compose
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "Error: Ni 'docker compose' ni 'docker-compose' están disponibles." >&2
    exit 1
fi

# 4. Configuración del entorno (.env)
if [ ! -f .env ]; then
    cp .env.example .env
    # Auto-generar pepper y clave admin inicial si openssl está instalado
    if command -v openssl &> /dev/null; then
        PEPPER=$(openssl rand -hex 32)
        ADMIN_KEY="isk_$(openssl rand -hex 24)"
        sed -i "s/^API_KEY_PEPPER=.*/API_KEY_PEPPER=$PEPPER/" .env
        sed -i "s/^BOOTSTRAP_ADMIN_API_KEY=.*/BOOTSTRAP_ADMIN_API_KEY=$ADMIN_KEY/" .env
        echo "================================================================" >&2
        echo " Se autogeneró API_KEY_PEPPER y BOOTSTRAP_ADMIN_API_KEY en .env" >&2
        echo " Tu API Key administrativa inicial es: $ADMIN_KEY" >&2
        echo "================================================================" >&2
    fi
    echo "================================================================" >&2
    echo " Se creó el archivo .env a partir de .env.example." >&2
    echo " Antes de continuar, por favor editá .env y completá:" >&2
    echo "   - OPENAI_API_KEY=tu_clave_real" >&2
    echo "   - PINECONE_API_KEY=tu_clave_real" >&2
    echo " Luego ejecutá nuevamente: ./run.sh" >&2
    echo "================================================================" >&2
    exit 1
fi

# Validar que no queden placeholders de ejemplo en .env
if grep -q "sk-tu_clave_openai_aqui" .env || grep -q "pcsk-tu_clave_pinecone_aqui" .env; then
    echo "================================================================" >&2
    echo " AVISO: .env contiene claves de ejemplo ('sk-tu_clave_...')." >&2
    echo " Editá .env con tus credenciales reales de OpenAI y Pinecone" >&2
    echo " para que el sistema de IA pueda arrancar correctamente." >&2
    echo "================================================================" >&2
    exit 1
fi

PROFILE_ARGS=()
if [ "${1:-}" = "--with-phoenix" ]; then
    PROFILE_ARGS=(--profile with-phoenix)
fi

echo "Iniciando despliegue con Docker Compose..."
$DOCKER_COMPOSE "${PROFILE_ARGS[@]}" up --build -d

echo ""
echo "================================================================"
echo " Sistema iniciado exitosamente."
echo " - API Docs (Swagger): http://localhost:8000/docs"
echo " - Health Check:       http://localhost:8000/api/v1/health"
if [ "${1:-}" = "--with-phoenix" ]; then
    echo " - Arize Phoenix UI:   http://localhost:6006"
fi
echo "================================================================"
$DOCKER_COMPOSE ps
