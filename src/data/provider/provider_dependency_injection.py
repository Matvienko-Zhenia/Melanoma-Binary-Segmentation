import os
from pathlib import Path

from dotenv import load_dotenv
from dependency_injector import containers, providers


from src.data.provider.ph2_provider import PH2Provider
from src.data.provider.gdrive_weights import WeightsProvider


load_dotenv()


class ProviderContainer(containers.DeclarativeContainer):
    """  
    A container class for managing provider instances.  

    This class uses dependency injection to create singleton instances of data providers.  

    :ivar ph2_image_provider: Singleton instance of the PH2Provider for handling PH2 dataset images.  
    :vartype ph2_image_provider: PH2Provider  
    :ivar weights_provider: Singleton instance of the WeightsProvider for downloading model weights.  
    :vartype weights_provider: WeightsProvider  
    """  
    ph2_image_provider = providers.Singleton(
        PH2Provider,
        dataset_path=os.getenv("LAB_DATASET_PATH", "dataset/PH2Dataset"),
    )

    ## TODO: not sure if we realy need it here 
    weights_provider = providers.Singleton(
        WeightsProvider,
        file_id=os.getenv("SEGNET_WEIGHTS_GID"),
        weights_path=os.getenv("LAB_WEIGHTS_PATH", "/tmp/"),
    )