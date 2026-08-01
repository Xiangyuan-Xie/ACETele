from __future__ import annotations

import sys
from pathlib import Path

package_root = Path(__file__).resolve().parents[1]
repository_root = Path(__file__).resolve().parents[3]
for path in (package_root, repository_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
