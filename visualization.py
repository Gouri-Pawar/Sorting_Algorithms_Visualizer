import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_sorting_visualization(steps, algorithm_name):
    if not steps:
        return None
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[f"{algorithm_name} - Step by Step Visualization"])
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
            data=[go.Bar(x=list(range(len(arr))),
                         y=arr,
                         marker=dict(color=colors),
                         text=arr,
                         textposition='auto')],
            name=str(i)
        ))
    fig.add_trace(go.Bar(x=list(range(len(steps[0][0]))),
                         y=steps[0][0],
                         marker=dict(color='lightblue'),
                         text=steps[0][0],
                         textposition='auto'))
    fig.frames = frames
    fig.update_layout(
        title=f"{algorithm_name} Sorting Animation",
        xaxis_title="Array Index", yaxis_title="Value", showlegend=False,
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True}],
                 "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}],
                 "label": "Pause", "method": "animate"}
            ],
            "direction": "left", "x": 0.1, "y": 0, "showactive": False, "type": "buttons"
        }]
    )
    return fig
