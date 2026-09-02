from src.config import PROJECT_ROOT, load_config


def test_load_config_default_resolves_against_project_root():
    config = load_config()
    assert config["random_forest"]["random_state"] == 42
    assert config["data"]["raw_path"] == "data/raw/bank-additional-full.csv"


def test_load_config_relative_path_ignores_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("config.yaml")
    assert "random_forest" in config


def test_load_config_absolute_path(tmp_path):
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text("foo:\n  bar: 1\n")
    config = load_config(custom_path)
    assert config == {"foo": {"bar": 1}}


def test_project_root_is_repo_root():
    assert (PROJECT_ROOT / "config.yaml").exists()
    assert (PROJECT_ROOT / "src").is_dir()
