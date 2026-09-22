#!/bin/sh
set -e

COUNTRY="${LEGALIZE_COUNTRY_CODE:-ar}"
REPO_DIR="${LEGALIZE_REPO_PATH:-repo_legalize_${COUNTRY}}"

case "$REPO_DIR" in
    /*) REPO_PATH="$REPO_DIR" ;;
    *)  REPO_PATH="/app/$REPO_DIR" ;;
esac

# Determinar URL del repositorio: si LEGALIZE_REPO_URL está definida se usa tal cual,
# de lo contrario se autoderiva a https://github.com/legalize-dev/legalize-${COUNTRY}.git
DEFAULT_REPO_URL="https://github.com/legalize-dev/legalize-${COUNTRY}.git"
REPO_URL="${LEGALIZE_REPO_URL:-$DEFAULT_REPO_URL}"

# Asegurar directorio y permisos adecuados
mkdir -p "$REPO_PATH" 2>/dev/null || true

# Si no existe .git en el volumen montado, clonar por única vez (con profundidad optimizada)
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "[entrypoint] Repositorio no encontrado en $REPO_PATH. Clonando $REPO_URL (depth 50)..."
    git clone --depth 50 "$REPO_URL" "$REPO_PATH"
else
    echo "[entrypoint] Repositorio detectado en $REPO_PATH. Sincronizando con git pull..."
    git -C "$REPO_PATH" pull --ff-only 2>/dev/null || echo "[entrypoint] Aviso: git pull completado o repositorio al día."
fi

# Acelerar consultas git (rev-list, log) con grafo de commits binario
git -C "$REPO_PATH" config core.commitGraph true 2>/dev/null || true
git -C "$REPO_PATH" commit-graph write --reachable 2>/dev/null || true

exec "$@"

