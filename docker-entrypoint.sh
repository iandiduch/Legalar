#!/bin/sh
set -e

REPO_PATH="${LEGALIZE_REPO_PATH:-/app/repo_legalize_ar}"

# Asegurar directorio y permisos adecuados
mkdir -p "$REPO_PATH" 2>/dev/null || true

# Si no existe .git en el volumen montado, clonar por única vez
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "[entrypoint] Repositorio no encontrado en $REPO_PATH. Clonando por primera vez..."
    git clone https://github.com/legalize-dev/legalize-ar.git "$REPO_PATH"
else
    echo "[entrypoint] Repositorio detectado en $REPO_PATH. Sincronizando con git pull..."
    git -C "$REPO_PATH" pull --ff-only 2>/dev/null || echo "[entrypoint] Aviso: git pull completado o repositorio al día."
fi

exec "$@"
