# Activate the thesis virtual environment and select the pinned native
# liboqs installation used for the final experiments.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: not inside the fl_pqc_thesis Git repository."
    return 1
fi

VENV_PATH="$REPO_ROOT/thesis_fl/venv"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "ERROR: virtual environment not found at:"
    echo "  $VENV_PATH"
    return 1
fi

source "$VENV_PATH/bin/activate"

export OQS_INSTALL_PATH="$VIRTUAL_ENV/oqs-0.16.0"

if [ ! -f "$OQS_INSTALL_PATH/lib/liboqs.so" ]; then
    echo "ERROR: liboqs 0.16.0 installation not found at:"
    echo "  $OQS_INSTALL_PATH"
    return 1
fi

# Avoid adding the same directory repeatedly if this file is sourced twice.
case ":${LD_LIBRARY_PATH:-}:" in
    *":$OQS_INSTALL_PATH/lib:"*)
        ;;
    *)
        export LD_LIBRARY_PATH="$OQS_INSTALL_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ;;
esac

python - <<'PY'
import sys
import oqs

expected = "0.16.0"
native = oqs.oqs_version()
wrapper = oqs.oqs_python_version()

print(f"Python:          {sys.executable}")
print(f"liboqs:          {native}")
print(f"liboqs-python:   {wrapper}")

if native != expected or wrapper != expected:
    raise SystemExit(
        f"ERROR: expected liboqs/liboqs-python {expected}, "
        f"got {native}/{wrapper}"
    )

print("OQS environment: OK")
PY
