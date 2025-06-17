import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from pyspark.core.broadcast import Broadcast
from pyspark.core.context import SparkContext


@contextmanager
def save_and_broadcast_model(
    model, rdd_context: SparkContext
) -> Generator[Broadcast[str], None, None]:
    with tempfile.TemporaryDirectory() as temp_base:
        unique_model_dir = Path(temp_base) / f"model_{uuid.uuid4().hex}"
        unique_model_dir.mkdir()
        model.save_pretrained(unique_model_dir)
        rdd_context.addFile(str(unique_model_dir), recursive=True)
        broadcast_dir = rdd_context.broadcast(unique_model_dir.name)
        yield broadcast_dir
        shutil.rmtree(unique_model_dir, ignore_errors=True)
