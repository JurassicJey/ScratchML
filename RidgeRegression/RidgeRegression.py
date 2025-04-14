import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, learning_rate = 0.0001, penalty = 0.05, stopping_threshold = 1e-9, iterations = 1000):
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.stopping_threshold = stopping_threshold
        self.iterations = iterations
        self.current_weight = 0
        self.current_bias = 0
        self.loss_history = []
    def fit(self, X, y):
        for iteration in range(self.iterations):
            predicted_y = self.current_weight @ X + self.current_bias
            ridge_loss = self.compute_ridge_loss(y, predicted_y) 
            

            if len(self.loss_history) > 0 and self.loss_history[iteration - 1] - self.loss_history[iteration] < self.stopping_threshold:
                print(f"training_loop stopped at iteration {iteration} and loss{ridge_loss:2f}")
                break
            

            
            print(f"loop: {iteration} current_loss: {ridge_loss}")
    def predict(self):
        pass
    def compute_ridge_loss(self, y_true, y_pred):
        residual_squared_sum = np.sum(np.square(y_true-y_pred))
        ridge_loss = residual_squared_sum + np.abs(self.current_weight) * self.penalty
        return ridge_loss
