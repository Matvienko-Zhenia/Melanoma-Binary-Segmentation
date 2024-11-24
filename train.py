import os
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim

from src.data.provider.gdrive_weights import WeightsProvider
from src.data.provider.ph2_provider import PH2Provider
from src.nn_models.nn_dependency_injection import NNContainer
from src.data.provider.provider_dependency_injection import ProviderContainer
from src.trainer.trainer_dependency_injection import TrainerContainer
from src.metrics.metrics_dependency_injection import MetricsContainer
from src.metrics.losses.bce_loss import BCELoss
from src.metrics.losses.dice_loss import DiceLoss
from src.metrics.losses.focal_loss import FocalLoss
from src.metrics.quality.iou import IOU

from src.utils.get_model import get_model_from_container

from src.trainer.trainer import Trainer

from dependency_injector.wiring import (
    inject,
    Provide,
)
from src.data.provider.ph2_provider import PH2Provider
from src.data.provider.gdrive_weights import WeightsProvider
from dotenv import load_dotenv


load_dotenv()


@inject
def get_container(global_container: NNContainer = Provide[NNContainer]):
    return global_container


class TrainModel:
    def __init__(self) -> None:
       pass 
    
    @inject
    def train_model(
        self,
        ph2_image_provider : PH2Provider = Provide[ProviderContainer.ph2_image_provider],
        weights_provider : WeightsProvider = Provide[ProviderContainer.weights_provider],
        dice_loss : DiceLoss = Provide[MetricsContainer.default_loss],
        iou_metric : IOU = Provide[MetricsContainer.iou_metric],
        trainer : Trainer = Provide[TrainerContainer.default_trainer],
    ) -> None:
        global_container : NNContainer = get_container().provider()

        epochs = int(os.getenv("LAB_EPOCHS", 1))
        device = os.getenv("LAB_DEVICE", "cpu")
        lab_weights_path = os.getenv("LAB_WEIGHTS_PATH", "/tmp/")
        Path(lab_weights_path).mkdir(exist_ok=True, parents=True)
        lab_weights_path += "model.pkl"

        data_train, data_validation, data_test = ph2_image_provider.get_data()
        model = get_model_from_container(
            global_container=global_container,
            nn_type=os.getenv("LAB_MODEL_TO_TRAIN", "segnet"),
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        train_data = trainer.train(
            model=model,
            opt=optimizer,
            loss_fn=dice_loss,
            epochs=epochs,
            data_tr=data_train,
            data_val=data_validation,
            path_file=lab_weights_path,
            device=device,
            scheduler=scheduler
        )
        
        iou_score = trainer.score_model(
            model=train_data["model"],
            metric=iou_metric, 
            data=data_test,
            device=device,
        )
        train_data["score_metric"] = iou_score

        print(train_data)
        return train_data

if __name__ == "__main__":
    provider_container = ProviderContainer()
    trainer_contrainer = TrainerContainer()
    metrics_container = MetricsContainer()
    provider_container.wire(modules=[__name__])
    trainer_contrainer.wire(modules=[__name__])
    metrics_container.wire(modules=[__name__])

    tc = TrainModel()
    tc.train_model()