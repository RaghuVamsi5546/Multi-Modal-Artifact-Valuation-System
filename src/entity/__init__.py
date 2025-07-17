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

@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    validation_data_path: Path
    test_data_path: Path
    text_features_column: str
    tfidf_max_features: int
    tfidf_n_gram_range: tuple
    count_vec_max_features: int
    count_vec_ngram_range: tuple
    sentence_transformer_models: list
    text_preprocessor_artifacts_dir: Path