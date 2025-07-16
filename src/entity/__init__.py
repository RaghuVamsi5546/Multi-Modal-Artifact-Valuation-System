from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    local_file: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    local_file: Path
    validation_status_path: Path