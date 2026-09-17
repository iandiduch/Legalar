#!/bin/sh
set -e

REPO_PATH="${LEGALIZE_REPO_PATH:-/app/repo_legalize_ar}"

# Asegurar directorio y permisos adecuados
mkdir -p "$REPO_PATH" 2>/dev/null || true

# Si no existe .git en el volumen montado, clonar por única vez (con profundidad optimizada)
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "[entrypoint] Repositorio no encontrado en $REPO_PATH. Clonando (depth 50 para máxima velocidad)..."
    git clone --depth 50 https://github.com/legalize-dev/legalize-ar.git "$REPO_PATH"
else
    echo "[entrypoint] Repositorio detectado en $REPO_PATH. Sincronizando con git pull..."
    git -C "$REPO_PATH" pull --ff-only 2>/dev/null || echo "[entrypoint] Aviso: git pull completado o repositorio al día."
fi

# Acelerar consultas git (rev-list, log) con grafo de commits binario
git -C "$REPO_PATH" config core.commitGraph true 2>/dev/null || true
git -C "$REPO_PATH" commit-graph write --reachable 2>/dev/null || true

exec "$@"

