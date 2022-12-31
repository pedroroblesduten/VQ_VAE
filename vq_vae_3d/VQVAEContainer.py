from typing import List, Any, Dict
from torch import nn, optim, Tensor
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import torch


def ClassifierContainerFactory(
    ErrorAssert,
    Printer,
    EarlyStopping,
    Fold,
    Batch,
    Dataset,
    validModels,
    device,
):
    class ClassifierContainer:
        def __init__(
            self,
            model: str,
            modelArgs: Dict = {},
            criterion: Any = nn.CrossEntropyLoss,
            criterionArgs: Dict = {},
            optimizer: Any = optim.Adam,
            optimizerArgs: Dict = {},
            nEpochs: int = 100,
            earlyStopping: bool = True,
            verbose: bool = True,
            patience: int = 7,
        ) -> None:
            ErrorAssert.valueAssert(
                (model is not None) and (len(model) > 0), "Model is required."
            )
            ErrorAssert.valueAssert(model in validModels, "Invalid model.")
            ErrorAssert.valueAssert(criterion is not None, "Invalid loss function.")
            ErrorAssert.valueAssert(optimizer is not None, "Invalid optimizer.")
            ErrorAssert.typeAssert(
                isinstance(nEpochs, int), "Number of epochs must be an int."
            )
            ErrorAssert.valueAssert(
                nEpochs > 0, "Number of epochs must be greater than 0."
            )
            ErrorAssert.typeAssert(
                earlyStopping is None or isinstance(patience, int),
                "Patience must be an int.",
            )

            self.__modelName: str = model
            self.__model: torch.nn = validModels[self.__modelName](**modelArgs).to(
                device=device
            )
            self.__criterion: torch.nn = criterion(**criterionArgs)
            self.__optimizer: torch.optim = optimizer(
                params=self.__model.parameters(),
                **optimizerArgs,
            )
            self.__nEpochs: int = nEpochs
            self.__earlyStopping: bool = earlyStopping
            self.__patience: int = patience
            self.__verbose: bool = verbose

        def initializeWeigths(self) -> None:
            self.__model.apply(self.__model.initializeWeights)

        def loadState(self, statePath: str) -> None:
            ErrorAssert(
                (statePath is not None) and (len(statePath) > 0),
                "Invalid state path.",
            )

            Printer.print(
                f"Loading model {self.__modelName}",
                disable=(not self.__verbose),
                fg="green",
                bold=True,
            )

            self.__model.load_state_dict(state_dict=torch.load(statePath))

        def saveState(self, statePath: str) -> None:
            ErrorAssert(
                (statePath is not None) and (len(statePath) > 0),
                "Invalid state path.",
            )

            Printer.print(
                f"Saving model {self.__modelName} state",
                disable=(not self.__verbose),
                fg="green",
                bold=True,
            )

            torch.save(obj=self.__model.state_dict(), f=statePath)

            
        def train(self, dataset: Dataset) -> None:
            ErrorAssert.typeAssert(isinstance(dataset, Dataset), "Invalid dataset.")
            ErrorAssert.valueAssert(
                dataset.withLabels(), "Dataset must have labels for classification."
            )

            Printer.print(
                f"Training model {self.__modelName} with dataset {dataset.getName()}"
                if dataset.getName()
                else f"Training model {self.__modelName}",
                disable=(not self.__verbose),
                fg="blue",
                bold=True,
            )

            trainSet: List[Batch] = dataset.getTrainSet()
            validationSet: List[Batch] = dataset.getValidationSet()

            totalTrainLosses: List[float] = []
            totalValidationLosses: List[float] = []

            totalValidationPredictions: List[int] = []
            totalValidationTrue: List[int] = []

            earlyStopping: EarlyStopping = (
                EarlyStopping(
                    patience=self.__patience,
                    verbose=False,
                )
                if self.__earlyStopping
                else None
            )

            nIters: int = (
                dataset.getTrainSize() * self.__nEpochs
                + dataset.getValidationSize() * self.__nEpochs
            )

            with tqdm(
                total=nIters,
                colour="green",
                disable=(not self.__verbose),
            ) as bar:
                for epoch in range(self.__nEpochs):
                    trainLosses: List[float] = []
                    validationLosses: List[float] = []

                    validationPredictions: List[int] = []
                    validationTrue: List[int] = []

                    self.__model.train()
                    for batch in trainSet:
                        trainData = batch.getData()

                        self.__optimizer.zero_grad()

                        decoded_images, min_indices, q_loss = self.__model(trainData)

                        rec_loss = self.__criterion(trainData, decoded_images)
                        vq_loss = rec_loss + q_loss
                        vq_loss.backward(retain_graph=True)
                        self.__optimizer.step()

                        # if epoch % 10 == 0:
                        #    decoded_images = 

                        bar.update(1)

                    self.__model.eval()
                    for batch in validationSet:

                       

            return totalTrainLosses, totalValidationLosses, confusionMatrix

        def predict(self, data: Tensor) -> np.ndarray:
            self.__model.eval()
            yPred: Tensor = self.__model(data)
            label = (torch.max(torch.exp(yPred), 1)[1]).data.cpu().numpy()

            return label

    return ClassifierContainer
