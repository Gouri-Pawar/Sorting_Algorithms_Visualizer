import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Smart Sorting Visualizer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .algorithm-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SortingAlgorithms:
    def _init_(self):
        self.comparisons = 0
        self.swaps = 0
        
    def reset_stats(self):
        self.comparisons = 0
        self.swaps = 0
    
    def bubble_sort(self, arr, visualize=False):
        self.reset_stats()
        arr = arr.copy()
        n = len(arr)
        steps = []
        
        for i in range(n):
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
                    if visualize:
                        steps.append((arr.copy(), j, j + 1, "swap"))
                elif visualize:
                    steps.append((arr.copy(), j, j + 1, "compare"))
        
        return arr, steps
    
    def selection_sort(self, arr, visualize=False):
        self.reset_stats()
        arr = arr.copy()
        n = len(arr)
        steps = []
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.comparisons += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
                if visualize:
                    steps.append((arr.copy(), i, j, "compare"))
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self.swaps += 1
                if visualize:
                    steps.append((arr.copy(), i, min_idx, "swap"))
        
        return arr, steps
    
    def insertion_sort(self, arr, visualize=False):
        self.reset_stats()
        arr = arr.copy()
        steps = []
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            while j >= 0 and arr[j] > key:
                self.comparisons += 1
                arr[j + 1] = arr[j]
                self.swaps += 1
                j -= 1
                if visualize:
                    steps.append((arr.copy(), j + 1, j + 2, "swap"))
            
            arr[j + 1] = key
            if visualize:
                steps.append((arr.copy(), j + 1, i, "insert"))
        
        return arr, steps
    
    def quick_sort(self, arr, visualize=False, steps=None, low=0, high=None):
        if steps is None:
            self.reset_stats()
            steps = []
            arr = arr.copy()
        
        if high is None:
            high = len(arr) - 1
        
        if low < high:
            pi = self._partition(arr, low, high, visualize, steps)
            self.quick_sort(arr, visualize, steps, low, pi - 1)
            self.quick_sort(arr, visualize, steps, pi + 1, high)
        
        return arr, steps
    
    def _partition(self, arr, low, high, visualize, steps):
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            self.comparisons += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.swaps += 1
                if visualize:
                    steps.append((arr.copy(), i, j, "swap"))
            elif visualize:
                steps.append((arr.copy(), j, high, "compare"))
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.swaps += 1
        if visualize:
            steps.append((arr.copy(), i + 1, high, "swap"))
        
        return i + 1

class MLPerformancePredictor:
    def _init_(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = ['array_size', 'initial_sortedness', 'value_range']
    
    def calculate_sortedness(self, arr):
        """Calculate how sorted an array is (0 = reverse sorted, 1 = fully sorted)"""
        if len(arr) <= 1:
            return 1.0
        
        inversions = 0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    inversions += 1
        
        max_inversions = n * (n - 1) / 2
        sortedness = 1 - (inversions / max_inversions) if max_inversions > 0 else 1
        return sortedness
    
    def generate_training_data(self, num_samples=1000):
        """Generate training data for ML model"""
        sorter = SortingAlgorithms()
        data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_samples):
            # Generate random array
            size = random.randint(10, 100)
            arr = [random.randint(1, 1000) for _ in range(size)]
            
            # Calculate features
            sortedness = self.calculate_sortedness(arr)
            value_range = max(arr) - min(arr)
            
            # Test each algorithm
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
        """Train ML model to predict performance"""
        # Create separate models for different metrics
        self.models = {}
        self.scalers = {}
        
        for target in ['execution_time', 'comparisons', 'swaps']:
            # Prepare features
            X = []
            y = []
            
            for algo in data['algorithm'].unique():
                algo_data = data[data['algorithm'] == algo]
                X_algo = algo_data[self.feature_names].values
                y_algo = algo_data[target].values
                
                # Add algorithm encoding (one-hot)
                algo_encoding = [0] * 4
                algo_idx = ['bubble_sort', 'selection_sort', 'insertion_sort', 'quick_sort'].index(algo)
                algo_encoding[algo_idx] = 1
                
                for i, features in enumerate(X_algo):
                    X.append(list(features) + algo_encoding)
                    y.append(y_algo[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            self.models[target] = model
        
        self.is_trained = True
    
    def predict_performance(self, array_size, sortedness, value_range, algorithm):
        """Predict performance metrics for given parameters"""
        if not self.is_trained:
            return None
        
        algo_encoding = [0] * 4
        algo_idx = ['bubble_sort', 'selection_sort', 'insertion_sort', 'quick_sort'].index(algorithm)
        algo_encoding[algo_idx] = 1
        
        features = [array_size, sortedness, value_range] + algo_encoding
        features = np.array(features).reshape(1, -1)
        
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(features)[0]
        
        return predictions

def create_sorting_visualization(steps, algorithm_name):
    """Create animated visualization of sorting steps"""
    if not steps:
        return None
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"{algorithm_name} - Step by Step Visualization"]
    )
    
    # Create frames for animation
    frames = []
    for i, (arr, idx1, idx2, action) in enumerate(steps):
        colors = ['lightblue'] * len(arr)
        if action == "swap":
            colors[idx1] = 'red'
            colors[idx2] = 'red'
        elif action == "compare":
            colors[idx1] = 'yellow'
            colors[idx2] = 'yellow'
        elif action == "insert":
            colors[idx1] = 'green'
        
        frames.append(go.Frame(
            data=[go.Bar(
                x=list(range(len(arr))),
                y=arr,
                marker=dict(color=colors),
                text=arr,
                textposition='auto'
            )],
            name=str(i)
        ))
    
    # Add initial frame
    fig.add_trace(go.Bar(
        x=list(range(len(steps[0][0]))),
        y=steps[0][0],
        marker=dict(color='lightblue'),
        text=steps[0][0],
        textposition='auto'
    ))
    
    fig.frames = frames
    
    fig.update_layout(
        title=f"{algorithm_name} Sorting Animation",
        xaxis_title="Array Index",
        yaxis_title="Value",
        showlegend=False,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🤖 Smart Sorting Algorithm Visualizer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application combines sorting algorithm visualization with machine learning to predict 
    performance metrics and provide intelligent recommendations.
    """)
    
    # Initialize session state
    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = MLPerformancePredictor()
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    
    # Sidebar configuration
    st.sidebar.header("🎛 Configuration")
    
    # Array configuration
    st.sidebar.subheader("Array Settings")
    array_size = st.sidebar.slider("Array Size", 5, 50, 15)
    generation_method = st.sidebar.selectbox(
        "Array Generation",
        ["Random", "Nearly Sorted", "Reverse Sorted", "Custom"]
    )
    
    # Generate array based on method
    if generation_method == "Random":
        if st.sidebar.button("Generate Random Array"):
            st.session_state.array = [random.randint(1, 100) for _ in range(array_size)]
    elif generation_method == "Nearly Sorted":
        if st.sidebar.button("Generate Nearly Sorted Array"):
            arr = list(range(1, array_size + 1))
            # Shuffle only 20% of elements
            shuffle_count = max(1, array_size // 5)
            for _ in range(shuffle_count):
                i, j = random.sample(range(array_size), 2)
                arr[i], arr[j] = arr[j], arr[i]
            st.session_state.array = arr
    elif generation_method == "Reverse Sorted":
        if st.sidebar.button("Generate Reverse Sorted Array"):
            st.session_state.array = list(range(array_size, 0, -1))
    else:  # Custom
        custom_input = st.sidebar.text_input(
            "Enter comma-separated values",
            "5,2,8,1,9,3"
        )
        if st.sidebar.button("Use Custom Array"):
            try:
                st.session_state.array = [int(x.strip()) for x in custom_input.split(',')]
            except ValueError:
                st.sidebar.error("Please enter valid integers separated by commas")
    
    # Initialize default array if not exists
    if 'array' not in st.session_state:
        st.session_state.array = [random.randint(1, 100) for _ in range(15)]
    
    # Algorithm selection
    st.sidebar.subheader("Algorithm Selection")
    algorithms = {
        "Bubble Sort": "bubble_sort",
        "Selection Sort": "selection_sort", 
        "Insertion Sort": "insertion_sort",
        "Quick Sort": "quick_sort"
    }
    
    selected_algorithm = st.sidebar.selectbox(
        "Choose Sorting Algorithm",
        list(algorithms.keys())
    )
    
    # ML Section
    st.sidebar.subheader("🤖 Machine Learning")
    if st.sidebar.button("Train ML Model"):
        with st.spinner("Training ML model... This may take a few minutes."):
            training_data = st.session_state.ml_predictor.generate_training_data(500)
            st.session_state.ml_predictor.train_model(training_data)
            st.session_state.training_data = training_data
            st.sidebar.success("Model trained successfully!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Current Array")
        # Display current array
        fig_array = go.Figure(data=[
            go.Bar(x=list(range(len(st.session_state.array))), 
                   y=st.session_state.array,
                   marker=dict(color='lightblue'),
                   text=st.session_state.array,
                   textposition='auto')
        ])
        fig_array.update_layout(
            title="Current Array to Sort",
            xaxis_title="Index",
            yaxis_title="Value",
            showlegend=False
        )
        st.plotly_chart(fig_array, use_container_width=True)
    
    with col2:
        st.subheader("📈 Array Statistics")
        arr = st.session_state.array
        stats_data = {
            "Size": len(arr),
            "Min Value": min(arr),
            "Max Value": max(arr),
            "Range": max(arr) - min(arr),
            "Mean": round(np.mean(arr), 2),
            "Std Dev": round(np.std(arr), 2)
        }
        
        for key, value in stats_data.items():
            st.metric(key, value)
        
        # Calculate sortedness
        sortedness = st.session_state.ml_predictor.calculate_sortedness(arr)
        st.metric("Sortedness", f"{sortedness:.2%}")
    
    # Sorting and Visualization
    if st.button("🚀 Sort and Visualize", type="primary"):
        sorter = SortingAlgorithms()
        algo_func = getattr(sorter, algorithms[selected_algorithm])
        
        # Perform sorting with visualization
        with st.spinner(f"Sorting with {selected_algorithm}..."):
            start_time = time.time()
            sorted_array, steps = algo_func(st.session_state.array.copy(), visualize=True)
            execution_time = time.time() - start_time
        
        # Display results
        st.subheader("📋 Sorting Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Execution Time", f"{execution_time:.4f}s")
        with col2:
            st.metric("Comparisons", sorter.comparisons)
        with col3:
            st.metric("Swaps", sorter.swaps)
        with col4:
            st.metric("Steps", len(steps))
        
        # Visualization
        st.subheader("🎬 Sorting Animation")
        fig_animation = create_sorting_visualization(steps, selected_algorithm)
        if fig_animation:
            st.plotly_chart(fig_animation, use_container_width=True)
        
        # Final result
        st.subheader("✅ Sorted Array")
        fig_sorted = go.Figure(data=[
            go.Bar(x=list(range(len(sorted_array))), 
                   y=sorted_array,
                   marker=dict(color='lightgreen'),
                   text=sorted_array,
                   textposition='auto')
        ])
        fig_sorted.update_layout(
            title="Final Sorted Array",
            xaxis_title="Index",
            yaxis_title="Value",
            showlegend=False
        )
        st.plotly_chart(fig_sorted, use_container_width=True)
    
    # ML Predictions Section
    if st.session_state.ml_predictor.is_trained:
        st.subheader("🔮 ML Performance Predictions")
        
        arr = st.session_state.array
        sortedness = st.session_state.ml_predictor.calculate_sortedness(arr)
        value_range = max(arr) - min(arr) if len(arr) > 0 else 0
        
        # Predict for all algorithms
        predictions_data = []
        for algo_name, algo_key in algorithms.items():
            pred = st.session_state.ml_predictor.predict_performance(
                len(arr), sortedness, value_range, algo_key
            )
            if pred:
                predictions_data.append({
                    'Algorithm': algo_name,
                    'Predicted Time (s)': f"{pred['execution_time']:.4f}",
                    'Predicted Comparisons': int(pred['comparisons']),
                    'Predicted Swaps': int(pred['swaps'])
                })
        
        if predictions_data:
            pred_df = pd.DataFrame(predictions_data)
            st.dataframe(pred_df, use_container_width=True)
            
            # Recommendation
            best_time_algo = min(predictions_data, 
                               key=lambda x: float(x['Predicted Time (s)']))
            st.success(f"🎯 *Recommended Algorithm*: {best_time_algo['Algorithm']} "
                      f"(Predicted time: {best_time_algo['Predicted Time (s)']}s)")
    
    # Algorithm Information
    st.subheader("📚 Algorithm Information")
    algo_info = {
        "Bubble Sort": {
            "Time Complexity": "O(n²)",
            "Space Complexity": "O(1)",
            "Description": "Repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.",
            "Best Case": "O(n)",
            "Stable": "Yes"
        },
        "Selection Sort": {
            "Time Complexity": "O(n²)",
            "Space Complexity": "O(1)",
            "Description": "Finds the minimum element and places it at the beginning, then repeats for the remaining elements.",
            "Best Case": "O(n²)",
            "Stable": "No"
        },
        "Insertion Sort": {
            "Time Complexity": "O(n²)",
            "Space Complexity": "O(1)",
            "Description": "Builds the final sorted array one item at a time by repeatedly taking an element and inserting it into its correct position.",
            "Best Case": "O(n)",
            "Stable": "Yes"
        },
        "Quick Sort": {
            "Time Complexity": "O(n log n)",
            "Space Complexity": "O(log n)",
            "Description": "Divides the array into smaller sub-arrays around a pivot element, then recursively sorts the sub-arrays.",
            "Best Case": "O(n log n)",
            "Stable": "No"
        }
    }
    
    info = algo_info[selected_algorithm]
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"*Algorithm*: {selected_algorithm}")
        st.markdown(f"*Time Complexity*: {info['Time Complexity']}")
        st.markdown(f"*Space Complexity*: {info['Space Complexity']}")
        st.markdown(f"*Best Case*: {info['Best Case']}")
        st.markdown(f"*Stable*: {info['Stable']}")
    
    with col2:
        st.markdown(f"*Description*: {info['Description']}")
    
    # Training Data Visualization
    if st.session_state.training_data is not None:
        with st.expander("📊 ML Training Data Analysis"):
            data = st.session_state.training_data
            
            # Performance comparison chart
            fig_perf = px.box(data, x='algorithm', y='execution_time',
                             title='Execution Time Distribution by Algorithm')
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Correlation with array size
            fig_corr = px.scatter(data, x='array_size', y='execution_time', 
                                 color='algorithm',
                                 title='Execution Time vs Array Size')
            st.plotly_chart(fig_corr, use_container_width=True)

if __name__ == "__main__":
    main()