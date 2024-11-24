from dotenv import load_dotenv
from fastapi import FastAPI

import src.endpoints.nn_inference as nn_inference
from src.nn_models.nn_dependency_injection import NNContainer


def create_app() -> FastAPI:
    """
    Initialise the FastAPI router with all existing routers
    :param app: app to initialise
    :return: FastAPI - Include router of app
    """
    app = FastAPI()
    app.include_router(nn_inference.router)

    return app


app = create_app()