import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, learning_rate = 0.0001, penalty = 0.05, stopping_threshold = 1e-9, iterations = 1000)
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.stopping_threshold = stopping_threshold
        self.iterations = iterations
        self.current_weight = 0
        self.current_bias = 0
        self.loss_history = []
    def fit(self, X, y):
        for iteration in self.iterations:

    def predict(self):
        pass

def compute_ridge_loss(y_true, y_pred):
    residual_squared_sum = np.square
