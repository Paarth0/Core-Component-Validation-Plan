import sys
import os
import pytest

# Get absolute path to src directory
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file)
project_root = os.path.dirname(tests_dir)
src_dir = os.path.join(project_root, 'src')

# Add to path if not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

print(f"[conftest] Added to sys.path: {src_dir}")

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# This decorator makes all async tests work automatically
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )