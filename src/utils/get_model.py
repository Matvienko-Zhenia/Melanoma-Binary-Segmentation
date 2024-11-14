import os

from src.nn_models.nn_dependency_injection import NNContainer


def get_model_from_container(global_container: NNContainer, nn_type: str):
    """  
    Retrieves a specified neural network model from a global container and loads its weights.

    This function checks the type of neural network specified and initializes the corresponding model 
    from the provided global container. It then loads pre-trained weights based on the model type 
    using environment variables.

    :param global_container: An instance of NNContainer that contains various model architectures.
    :type global_container: NNContainer
    :param nn_type: A string indicating the type of model to retrieve. Options include "segnet", "unet", or "naive".
    :type nn_type: str
    :returns: The initialized model with loaded weights.
    :rtype: nn.Module
    """  
    if nn_type == "segnet":
        model = global_container.segnet()
        model.load_from_provider(weights_name=os.getenv("SEGNET_WEIGHTS_NAME"))
    elif nn_type == "unet":
        model = global_container.unet()
        model.load_from_provider(weights_name=os.getenv("UNET_WEIGHTS_NAME"))
    elif nn_type == "naive":
        model = global_container.Naivecnn()
        model.load_from_provider(weights_name=os.getenv("NaiveCNN_WEIGHTS_NAME"))
    else:
        model = global_container.segnet()
        model.load_from_provider(weights_name=os.getenv("SEGNET_WEIGHTS_NAME"))

    model.eval()
    return model