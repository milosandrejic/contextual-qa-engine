from fastapi import FastAPI
from app.core.config import settings
from app.routers import upload

app = FastAPI(title=settings.app_name)

app.include_router(upload.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
