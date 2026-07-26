from pathlib import Path

from app.core.config import Settings


class SettingsService:
    def __init__(self, settings: Settings, env_path: Path | None = None) -> None:
        self.settings = settings
        self.env_path = env_path

    def huggingface_token_configured(self) -> bool:
        return bool(self.settings.huggingface_token)

    def save_huggingface_token(self, token: str | None) -> None:
        cleaned = (token or "").strip()
        if "\n" in (token or "") or "\r" in (token or ""):
            raise ValueError("Hugging Face token must be a single line")
        self._write_env_values(
            {
                "HUGGINGFACE_HUB_TOKEN": cleaned or None,
                "HF_TOKEN": None,
            }
        )

    def openrouter_key_configured(self) -> bool:
        return bool(self.settings.openrouter_key)

    def openrouter_model(self) -> str | None:
        model = (self.settings.openrouter_model or "").strip()
        return model or None

    def save_openrouter_key(self, key: str | None) -> None:
        cleaned = (key or "").strip()
        if "\n" in (key or "") or "\r" in (key or ""):
            raise ValueError("OpenRouter API key must be a single line")
        self._write_env_values({"OPENROUTER_API_KEY": cleaned or None})

    def save_openrouter_model(self, model: str | None) -> None:
        cleaned = (model or "").strip()
        if "\n" in (model or "") or "\r" in (model or ""):
            raise ValueError("OpenRouter model must be a single line")
        self._write_env_values({"OPENROUTER_MODEL": cleaned or None})

    def _write_env_values(self, updates: dict[str, str | None]) -> None:
        env_path = self.env_path or self.settings.repo_root / ".env"
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        next_lines: list[str] = []
        replaced: set[str] = set()
        for line in lines:
            stripped = line.strip()
            key = stripped.split("=", 1)[0] if "=" in stripped else ""
            if key in updates:
                replaced.add(key)
                if updates[key] is not None:
                    next_lines.append(f"{key}={updates[key]}")
                continue
            next_lines.append(line)
        for key, value in updates.items():
            if value is not None and key not in replaced:
                next_lines.append(f"{key}={value}")
        env_path.write_text(
            "\n".join(next_lines).rstrip() + ("\n" if next_lines else ""),
            encoding="utf-8",
        )
