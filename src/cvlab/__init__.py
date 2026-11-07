"""cvlab: shared, tested utilities for the computer-vision-for-robotics notebooks.

Submodules are imported explicitly so that importing :mod:`cvlab` stays cheap
(no torch import unless you ask for it)::

    from cvlab import device, geometry, io, metrics, pointcloud, viz
"""

__version__ = "0.1.0"

__all__ = [
    "bench",
    "device",
    "download",
    "geometry",
    "io",
    "metrics",
    "pointcloud",
    "viz",
]
