from fastapi import FastAPI
from app.core.config import settings
from app.routers import upload, search, ask, session, documents

app = FastAPI(title=settings.app_name)

app.include_router(upload.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(session.router)
app.include_router(documents.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
