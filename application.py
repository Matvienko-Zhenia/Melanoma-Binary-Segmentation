from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.endpoints.nn_inference as nn_inference
from src.nn_models.nn_dependency_injection import NNContainer


def create_app() -> FastAPI:
    """
    Initialise the FastAPI router with all existing routers
    :param app: app to initialise
    :return: FastAPI - Include router of app
    """
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],  # Vue's default dev server ports
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(nn_inference.router)

    return app


app = create_app()