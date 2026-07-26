from fastapi import APIRouter, HTTPException

from app.container import refresh_settings, settings_service
from app.schemas import PlatformSettingsRead, PlatformSettingsUpdate

router = APIRouter(prefix="/settings")


def _read() -> PlatformSettingsRead:
    return PlatformSettingsRead(
        huggingface_hub_token_configured=settings_service.huggingface_token_configured(),
        openrouter_api_key_configured=settings_service.openrouter_key_configured(),
        openrouter_model=settings_service.openrouter_model(),
    )


@router.get("", response_model=PlatformSettingsRead)
def get_platform_settings() -> PlatformSettingsRead:
    return _read()


@router.patch("", response_model=PlatformSettingsRead)
def update_platform_settings(payload: PlatformSettingsUpdate) -> PlatformSettingsRead:
    # Only touch fields the caller actually sent, so a partial update (e.g.
    # saving the OpenRouter model) never clears an unrelated secret.
    provided = payload.model_fields_set
    try:
        if "huggingface_hub_token" in provided:
            settings_service.save_huggingface_token(payload.huggingface_hub_token)
        if "openrouter_api_key" in provided:
            settings_service.save_openrouter_key(payload.openrouter_api_key)
        if "openrouter_model" in provided:
            settings_service.save_openrouter_model(payload.openrouter_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    refresh_settings()
    return _read()
