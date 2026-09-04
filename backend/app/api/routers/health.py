from fastapi import APIRouter

from app.cli import __version__

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness, plus enough to tell two builds apart.

    `orinth version` exists to answer "is the CLI I am running the same build as
    the server it is talking to", which it cannot do without this. Additive:
    phase 18's desktop supervisor polls this for a 200 and ignores the body.
    """
    return {"status": "ok", "app": "orinth", "version": __version__}
