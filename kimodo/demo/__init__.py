# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Ayn1631 in 2026.
# Changes: load .env configuration and support low-VRAM startup defaults.

# ruff: noqa: I001
import argparse
import dotenv
import os
from kimodo.model import DEFAULT_MODEL
from kimodo.model.registry import resolve_model_name

from .app import Demo


def main() -> None:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Run the kimodo demo UI.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Default model to load (e.g. Kimodo-SOMA-RP-v1, kimodo-soma-rp, or SOMA).",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable multi-tier memory offloading (Disk-RAM-VRAM) for low-memory GPUs.",
        default=os.getenv("CPU_Load", "False").lower() in ("true", "1", "t"),
    )
    args = parser.parse_args()

    resolved = resolve_model_name(args.model, "Kimodo")
    demo = Demo(default_model_name=resolved, offload=args.offload)
    demo.run()


if __name__ == "__main__":
    main()
