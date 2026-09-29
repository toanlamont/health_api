import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.api.auth_router import router as auth_router
from app.api.health_router import router as health_router

# uvicorn configures its own loggers and leaves the root one alone, so without
# this every log.info() in the app is swallowed at the default WARNING level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="HeartBeat API", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router)
app.include_router(health_router)

@app.get("/")
async def root():
    return {"message": "HeartBeat API is running"}
