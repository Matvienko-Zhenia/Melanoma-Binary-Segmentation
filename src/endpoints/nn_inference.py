import io
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import torch

from fastapi import (
    APIRouter,
    File, 
    UploadFile,
    Form,
)
from fastapi.responses import StreamingResponse  
from dependency_injector.wiring import (
    inject,
    Provide,
)
from skimage.io import imread
from skimage.transform import resize

from src.nn_models.nn_dependency_injection import NNContainer
from src.nn_models.segnet import SegNet
from src.nn_models.unet import UNet
from src.nn_models.naive_cnn import NaiveCnn
from src.utils.get_model import get_model_from_container
from src.utils.read_files import read_file

router = APIRouter()


logger = logging.getLogger('uvicorn.error')
logger.setLevel("INFO")


@inject
def get_container(global_container: NNContainer = Provide[NNContainer]):
    """  
    Retrieves the global NNContainer instance.  

    :param global_container: The dependency-injected NNContainer.  
    :type global_container: NNContainer  
    :returns: The global NNContainer instance.  
    :rtype: NNContainer  
    """  
    return global_container


def create_img(image, inference):
    """  
    Creates a PNG image showing the input image and its inference result.  

    :param image: The input image tensor.  
    :type image: np.ndarray  
    :param inference: The output of the neural network inference.  
    :type inference: np.ndarray  
    :returns: A byte buffer containing the saved figure.  
    :rtype: io.BytesIO  
    """  
    plt.figure()
    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(np.rollaxis(image, 1, 4).squeeze(0))
    axarr[1].imshow(np.rollaxis(inference, 1, 4).squeeze(0))
    buffer = io.BytesIO()
    plt.savefig(buffer)
    buffer.seek(0)
    plt.close(f)

    return buffer



@router.post("/nn_inference")
async def nn_inference(
    file: UploadFile = File(...),
    nn_type: str = Form(...),
):
    """  
    Handles the POST request for neural network inference.  

    Accepts an image file and the type of neural network to use, processes the image,  
    performs inference, and returns the result as a PNG image.  

    :param file: The uploaded image file.  
    :type file: UploadFile  
    :param nn_type: The type of neural network to use (e.g., SegNet, UNet).  
    :type nn_type: str  
    :returns: A StreamingResponse containing the PNG image of the input and inference result.  
    :rtype: StreamingResponse  
    """  
    global_container : NNContainer = get_container().provider()

    model = get_model_from_container(
        global_container=global_container,
        nn_type=str(nn_type),
    )
    file = file.file.read()

    image = resize(
        imread(io.BytesIO(file)), 
        (256, 256), 
        mode='constant',
        anti_aliasing=True
    )[:, :, :3]
    image = np.rollaxis(np.array(image)[np.newaxis, :], 3, 1)

    logger.info(str(image.shape))
    logger.info(type(model))


    inference = model(
        torch.from_numpy(
            image,
        ).to(torch.device('cpu'), dtype=torch.float32)
    )
    inference = inference.detach().numpy()
    logger.info(str(inference.shape))

    buffer = create_img(
        image, inference
    )
    
    return StreamingResponse(
        content=buffer,
        media_type="image/png",
    )