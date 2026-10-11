# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Entry-point shim so the documented command works:
``python -m mivia_small_json.train`` (see AGENTS.md commands table).

All logic lives in model.py (module contract); this file only forwards.
"""

from mivia_small_json.model import main

if __name__ == "__main__":
    main()
