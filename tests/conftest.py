import shutil
from pathlib import Path
import pytest


@pytest.fixture
def clean_tmp_dir(tmp_path):
    path = Path(tmp_path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    yield path

    if path.exists():
        shutil.rmtree(path)
