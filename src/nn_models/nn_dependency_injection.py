import os
from pathlib import Path

from dotenv import load_dotenv
from dependency_injector import containers, providers


from src.data.provider.gdrive_weights import WeightsProvider

# from src.nn_models.segnet import SegNet # TODO: implement!
from src.nn_models.unet import UNet
from src.nn_models.naive_cnn import NaiveCnn


load_dotenv()


class NNContainer(containers.DeclarativeContainer):
    """
    A container for neural network models and their associated weight providers.

    This class uses the `DeclarativeContainer` from the `dependency_injector` library to manage 
    the creation of various neural network models and their respective weight providers. 

    Attributes:
        wiring_config (containers.WiringConfiguration): 
            Configuration for wiring modules in the container.
        
        weights_provider_segnet (providers.Singleton): 
            Singleton provider for SegNet weights.

        weights_provider_unet (providers.Singleton): 
            Singleton provider for UNet weights.

        weights_provider_Naivecnn (providers.Singleton): 
            Singleton provider for Naive CNN weights.

        segnet (providers.Factory): 
            Factory for creating instances of the SegNet model.

        unet (providers.Factory): 
            Factory for creating instances of the UNet model.

        Naivecnn (providers.Factory): 
            Factory for creating instances of the Naive CNN model.

        default_nn (providers.Factory): 
            Default factory for creating a SegNet instance.
    """
    wiring_config = containers.WiringConfiguration(
        modules=[
            ".endpoints.nn_inference",
            "train"
        ]
    )
    
    # weights_provider_segnet = providers.Singleton(
    #     WeightsProvider,
    #     file_id=os.getenv("SEGNET_WEIGHTS_GID"),
    #     weights_path = Path(os.getenv("WEIGHTS_PATH_ROOT", "./artifacts/models"))
    # ) # TODO: implement!

    weights_provider_unet = providers.Singleton(
        WeightsProvider,
        file_id=os.getenv("UNET_WEIGHTS_GID"),
        weights_path = Path(os.getenv("WEIGHTS_PATH_ROOT", "./artifacts/models"))
    )

    weights_provider_Naivecnn = providers.Singleton(
        WeightsProvider,
        file_id=os.getenv("NaiveCNN_WEIGHTS_GID"),
        weights_path = Path(os.getenv("WEIGHTS_PATH_ROOT", "./artifacts/models"))
    )

    unet = providers.Factory(
        UNet,
        weights_provider=weights_provider_unet,
    )

    Naivecnn = providers.Factory(
        NaiveCnn,
        weights_provider=weights_provider_unet,
    )
    # segnet = providers.Factory(
    #     SegNet,
    #     weights_provider=weights_provider_segnet,
    # ) # TODO: implement!

    default_nn = providers.Factory(
        UNet,
        weights_provider=weights_provider_segnet,
    )