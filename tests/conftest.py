import sys
import os
import pytest

# Add the project root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure pytest-asyncio default behavior
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as asyncio")
