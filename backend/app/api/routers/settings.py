from fastapi import APIRouter, HTTPException

from app.container import refresh_settings, settings_service
from app.schemas import PlatformSettingsRead, PlatformSettingsUpdate

router = APIRouter(prefix="/settings")


@router.get("", response_model=PlatformSettingsRead)
def get_platform_settings() -> PlatformSettingsRead:
    return PlatformSettingsRead(
        huggingface_hub_token_configured=settings_service.huggingface_token_configured()
    )


@router.patch("", response_model=PlatformSettingsRead)
def update_platform_settings(payload: PlatformSettingsUpdate) -> PlatformSettingsRead:
    try:
        settings_service.save_huggingface_token(payload.huggingface_hub_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    refresh_settings()
    return PlatformSettingsRead(
        huggingface_hub_token_configured=settings_service.huggingface_token_configured()
    )
