import random, time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sorting_algorithms import SortingAlgorithms

class MLPerformancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = ['array_size', 'initial_sortedness', 'value_range']

    def calculate_sortedness(self, arr):
        if len(arr) <= 1:
            return 1.0
        inversions = 0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    inversions += 1
        max_inversions = n * (n - 1) / 2
        return 1 - (inversions / max_inversions) if max_inversions > 0 else 1

    def generate_training_data(self, num_samples=1000):
        sorter = SortingAlgorithms()
        data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(num_samples):
            size = random.randint(10, 100)
            arr = [random.randint(1, 1000) for _ in range(size)]
            sortedness = self.calculate_sortedness(arr)
            value_range = max(arr) - min(arr)
            algorithms = {
                'bubble_sort': sorter.bubble_sort,
                'selection_sort': sorter.selection_sort,
                'insertion_sort': sorter.insertion_sort,
                'quick_sort': sorter.quick_sort
            }
            for algo_name, algo_func in algorithms.items():
                start_time = time.time()
                _, _ = algo_func(arr.copy())
                execution_time = time.time() - start_time
                data.append({
                    'algorithm': algo_name,
                    'array_size': size,
                    'initial_sortedness': sortedness,
                    'value_range': value_range,
                    'comparisons': sorter.comparisons,
                    'swaps': sorter.swaps,
                    'execution_time': execution_time
                })
            progress_bar.progress((i + 1) / num_samples)
            status_text.text(f'Generating training data: {i + 1}/{num_samples}')
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame(data)

    def train_model(self, data):
        self.models = {}
        for target in ['execution_time', 'comparisons', 'swaps']:
            X, y = [], []
            for algo in data['algorithm'].unique():
                algo_data = data[data['algorithm'] == algo]
                X_algo = algo_data[self.feature_names].values
                y_algo = algo_data[target].values
                algo_encoding = [0] * 4
                algo_idx = ['bubble_sort', 'selection_sort', 'insertion_sort', 'quick_sort'].index(algo)
                algo_encoding[algo_idx] = 1
                for i, features in enumerate(X_algo):
                    X.append(list(features) + algo_encoding)
                    y.append(y_algo[i])
            X, y = np.array(X), np.array(y)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            self.models[target] = model
        self.is_trained = True

    def predict_performance(self, array_size, sortedness, value_range, algorithm):
        if not self.is_trained:
            return None
        algo_encoding = [0] * 4
        algo_idx = ['bubble_sort', 'selection_sort', 'insertion_sort', 'quick_sort'].index(algorithm)
        algo_encoding[algo_idx] = 1
        features = [array_size, sortedness, value_range] + algo_encoding
        features = np.array(features).reshape(1, -1)
        predictions = {target: model.predict(features)[0] for target, model in self.models.items()}
        return predictions
