"""Line profiler utilities.

This module defines a @profile decorator that can stand
in if line_profiler is not used.
"""
import builtins

try:
    profile = builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        """Stub replacing line profiler's profile decorator."""
        return func
