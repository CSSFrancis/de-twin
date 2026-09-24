"""Detector model: camera catalogue and the raw-frame noise pipeline."""

from .detector import Detector
from .models import CAMERAS, CameraModel, camera

__all__ = ["CAMERAS", "CameraModel", "Detector", "camera"]
