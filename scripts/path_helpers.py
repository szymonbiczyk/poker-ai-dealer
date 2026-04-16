from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_default_sample_image_path(filename: str = "card_test.jpg") -> Path:
    return get_data_dir() / "samples" / filename


def get_processed_dir(*, create: bool = False) -> Path:
    path = get_data_dir() / "processed"

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_templates_dir() -> Path:
    return get_data_dir() / "templates"


def get_rank_templates_dir() -> Path:
    return get_templates_dir() / "ranks"


def get_suit_templates_dir() -> Path:
    return get_templates_dir() / "suits"


def get_kaggle_cards_dir() -> Path:
    return get_data_dir() / "raw" / "kaggle_cards"
