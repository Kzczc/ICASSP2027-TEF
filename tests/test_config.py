import pytest

from tef.config import DEFAULT_MODEL_CONFIG, load_model_config


def test_environment_expansion(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text(
        "demo:\n"
        "  backend: vllm\n"
        "  model_path: ${DEMO_MODEL_PATH:-org/model}\n"
        "  base_url: ${DEMO_BASE_URL}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEMO_MODEL_PATH", raising=False)
    monkeypatch.setenv("DEMO_BASE_URL", "http://localhost:8000/v1")
    config = load_model_config("demo", path)
    assert config["model_path"] == "org/model"
    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["name"] == "demo"

    monkeypatch.setenv("DEMO_MODEL_PATH", "/models/demo")
    assert load_model_config("demo", path)["model_path"] == "/models/demo"


def test_unknown_model(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("demo:\n  backend: vllm\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_model_config("missing", path)


@pytest.mark.parametrize("name", ["qwen2.5-7b", "llama3-8b", "qwen3-14b", "deepseek-v3.2", "gpt-4o-mini"])
def test_paper_models_are_registered(name):
    config = load_model_config(name, DEFAULT_MODEL_CONFIG)
    assert config["backend"] in ("vllm", "openai")
