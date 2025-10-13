from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import router as auth_router
from .config import CORS_ORIGINS
from .routes import chat as chat_routes
from .routes import system as system_routes

logger = logging.getLogger("agrichat.app.factory")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] App initialized.")
    yield
    logger.info("[Shutdown] App shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    app.include_router(system_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(auth_router)

    return app
