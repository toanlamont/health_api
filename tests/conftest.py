"""Point the app at a throwaway database before it is imported.

app.config reads DATABASE_URL at import time, so this has to happen before any
test module pulls in the app — which is exactly what conftest is loaded for.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///" + os.path.join(
    tempfile.mkdtemp(), "test.db"
).replace("\\", "/")
os.environ.setdefault("SECRET_KEY", "test-secret")

# The endpoint archives every capture it is asked to store, so a test run would
# otherwise bury the real measurements in captures/ under its own synthetic ones.
os.environ["HEARTBEAT_SAVE_CAPTURES"] = "0"
