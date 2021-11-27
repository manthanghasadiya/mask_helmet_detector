#!/usr/bin/env python
import os
import sys
os.add_dll_directory(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mask_helmet_detector.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
