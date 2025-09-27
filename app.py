import streamlit as st
import random, time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from sorting_algorithms import SortingAlgorithms
from ml_predictor import MLPerformancePredictor
from visualization import create_sorting_visualization
from utils import calculate_array_stats


# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Smart Sorting Visualizer",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Main Application ---
def main():
    st.markdown('<h1 class="main-header">🤖 Smart Sorting Algorithm Visualizer</h1>', unsafe_allow_html=True)

    st.markdown("""
    This application combines sorting algorithm visualization with machine learning to 
    predict performance metrics and provide intelligent recommendations.
    """)

    # Session state
    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = MLPerformancePredictor()
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None

    # Sidebar
    st.sidebar.header("🎛 Configuration")

    # Array settings
    st.sidebar.subheader("Array Settings")
    array_size = st.sidebar.slider("Array Size", 5, 50, 15)
    generation_method = st.sidebar.selectbox(
        "Array Generation", ["Random", "Nearly Sorted", "Reverse Sorted", "Custom"]
    )

    if generation_method == "Random":
        if st.sidebar.button("Generate Random Array"):
            st.session_state.array = [random.randint(1, 100) for _ in range(array_size)]
    elif generation_method == "Nearly Sorted":
        if st.sidebar.button("Generate Nearly Sorted Array"):
            arr = list(range(1, array_size + 1))
            shuffle_count = max(1, array_size // 5)
            for _ in range(shuffle_count):
                i, j = random.sample(range(array_size), 2)
                arr[i], arr[j] = arr[j], arr[i]
            st.session_state.array = arr
    elif generation_method == "Reverse Sorted":
        if st.sidebar.button("Generate Reverse Sorted Array"):
            st.session_state.array = list(range(array_size, 0, -1))
    else:  # Custom
        custom_input = st.sidebar.text_input("Enter comma-separated values", "5,2,8,1,9,3")
        if st.sidebar.button("Use Custom Array"):
            try:
                st.session_state.array = [int(x.strip()) for x in custom_input.split(',')]
            except ValueError:
                st.sidebar.error("Please enter valid integers separated by commas")

    # Default array
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
    selected_algorithm = st.sidebar.selectbox("Choose Sorting Algorithm", list(algorithms.keys()))

    # ML training
    st.sidebar.subheader("🤖 Machine Learning")
    if st.sidebar.button("Train ML Model"):
        with st.spinner("Training ML model... This may take a few minutes."):
            training_data = st.session_state.ml_predictor.generate_training_data(500)
            st.session_state.ml_predictor.train_model(training_data)
            st.session_state.training_data = training_data
            st.sidebar.success("Model trained successfully!")

    # --- Main content ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Current Array")
        fig_array = go.Figure(data=[go.Bar(
            x=list(range(len(st.session_state.array))),
            y=st.session_state.array,
            marker=dict(color='lightblue'),
            text=st.session_state.array,
            textposition='auto'
        )])
        fig_array.update_layout(title="Current Array to Sort", xaxis_title="Index", yaxis_title="Value")
        st.plotly_chart(fig_array, use_container_width=True)

    with col2:
        st.subheader("📈 Array Statistics")
        stats_data = calculate_array_stats(st.session_state.array)
        for key, value in stats_data.items():
            st.metric(key, value)
        sortedness = st.session_state.ml_predictor.calculate_sortedness(st.session_state.array)
        st.metric("Sortedness", f"{sortedness:.2%}")

    # Sorting & Visualization
    if st.button("🚀 Sort and Visualize", type="primary"):
        sorter = SortingAlgorithms()
        algo_func = getattr(sorter, algorithms[selected_algorithm])
        with st.spinner(f"Sorting with {selected_algorithm}..."):
            start_time = time.time()
            sorted_array, steps = algo_func(st.session_state.array.copy(), visualize=True)
            execution_time = time.time() - start_time

        # Results
        st.subheader("📋 Sorting Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Execution Time", f"{execution_time:.4f}s")
        c2.metric("Comparisons", sorter.comparisons)
        c3.metric("Swaps", sorter.swaps)
        c4.metric("Steps", len(steps))

        # Animation
        st.subheader("🎬 Sorting Animation")
        fig_animation = create_sorting_visualization(steps, selected_algorithm)
        if fig_animation:
            st.plotly_chart(fig_animation, use_container_width=True)

        # Final sorted array
        st.subheader("✅ Sorted Array")
        fig_sorted = go.Figure(data=[go.Bar(
            x=list(range(len(sorted_array))),
            y=sorted_array,
            marker=dict(color='lightgreen'),
            text=sorted_array,
            textposition='auto'
        )])
        fig_sorted.update_layout(title="Final Sorted Array", xaxis_title="Index", yaxis_title="Value")
        st.plotly_chart(fig_sorted, use_container_width=True)

    # ML Predictions
    if st.session_state.ml_predictor.is_trained:
        st.subheader("🔮 ML Performance Predictions")
        arr = st.session_state.array
        sortedness = st.session_state.ml_predictor.calculate_sortedness(arr)
        value_range = max(arr) - min(arr) if arr else 0
        predictions_data = []
        for algo_name, algo_key in algorithms.items():
            pred = st.session_state.ml_predictor.predict_performance(len(arr), sortedness, value_range, algo_key)
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
            best_time_algo = min(predictions_data, key=lambda x: float(x['Predicted Time (s)']))
            st.success(f"🎯 *Recommended Algorithm*: {best_time_algo['Algorithm']} "
                       f"(Predicted time: {best_time_algo['Predicted Time (s)']}s)")

    # Algorithm info
    st.subheader("📚 Algorithm Information")
    algo_info = {
        "Bubble Sort": {"Time Complexity": "O(n²)", "Space Complexity": "O(1)", "Best Case": "O(n)", "Stable": "Yes",
                        "Description": "Repeatedly compares and swaps adjacent elements."},
        "Selection Sort": {"Time Complexity": "O(n²)", "Space Complexity": "O(1)", "Best Case": "O(n²)", "Stable": "No",
                           "Description": "Finds the minimum element and places it at the beginning."},
        "Insertion Sort": {"Time Complexity": "O(n²)", "Space Complexity": "O(1)", "Best Case": "O(n)", "Stable": "Yes",
                           "Description": "Builds the sorted array one item at a time."},
        "Quick Sort": {"Time Complexity": "O(n log n)", "Space Complexity": "O(log n)", "Best Case": "O(n log n)", "Stable": "No",
                       "Description": "Divides around a pivot and recursively sorts sub-arrays."}
    }
    info = algo_info[selected_algorithm]
    c1, c2 = st.columns(2)
    c1.markdown(f"*Algorithm*: {selected_algorithm}")
    c1.markdown(f"*Time Complexity*: {info['Time Complexity']}")
    c1.markdown(f"*Space Complexity*: {info['Space Complexity']}")
    c1.markdown(f"*Best Case*: {info['Best Case']}")
    c1.markdown(f"*Stable*: {info['Stable']}")
    c2.markdown(f"*Description*: {info['Description']}")

    # Training Data Analysis
    if st.session_state.training_data is not None:
        with st.expander("📊 ML Training Data Analysis"):
            data = st.session_state.training_data
            fig_perf = px.box(data, x='algorithm', y='execution_time',
                              title='Execution Time Distribution by Algorithm')
            st.plotly_chart(fig_perf, use_container_width=True)
            fig_corr = px.scatter(data, x='array_size', y='execution_time',
                                  color='algorithm',
                                  title='Execution Time vs Array Size')
            st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
