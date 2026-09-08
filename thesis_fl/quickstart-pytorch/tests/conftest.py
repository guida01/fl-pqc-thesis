import os
import sys

# Make `pytorchexample` importable without installing the package, same
# approach as the existing smoke_test.py at the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
