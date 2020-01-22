import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
import numpy as np
from numpy import ndarray
import pandas as pd


__all__ = ['Ensemble']


class DummyTransform:
    def __call__(self, input): return input


class Ensemble:
    """
    Abstraction for an image classifier. Support user defined test time augmentation
    """

    def __init__(self, model, transform_main=DummyTransform()):
        self._model = model
        self._model.eval()
        self.transform_main = transform_main

    @torch.no_grad()
    def predict(self, image: Tensor, return_prob: bool = False, tries: int = 5) -> ndarray:
        pass

    @torch.no_grad()
    def export_predictions(self, test_data: Dataset, path: str, tries: int = 5):
        pass
