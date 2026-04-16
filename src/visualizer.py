import numpy as np
import plotly.graph_objects as go

def plot_camera_trajectory(poses):
    if len(poses) < 2:
        return None
    positions = np.array([p[:3, 3] for p in poses])
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2],
                               mode='lines+markers', name='Camera path'))
    fig.update_layout(title="Camera Trajectory (SLAM)", scene=dict(aspectmode='data'))
    return fig

"""
def plot_risk_heatmap(dust_concentration):
    # Simplified – you can expand later
    fig = go.Figure(data=[go.Volume(
        x=np.arange(dust_concentration.shape[0]),
        y=np.arange(dust_concentration.shape[1]),
        z=np.arange(dust_concentration.shape[2]),
        value=dust_concentration.flatten(),
        isomin=0.1, isomax=0.8, opacity=0.1, surface_count=17, colorscale='Reds'
    )])
    return fig

    """

import numpy as np
import plotly.graph_objects as go

def plot_risk_heatmap(dust):
    nx, ny, nz = dust.shape
    
    # Create coordinate grid mapped to voxel indices
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten everything
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()
    Df = dust.flatten()
    
    # Show only meaningful dust values
    threshold = 0.2
    mask = Df > threshold
    
    fig = go.Figure(data=go.Scatter3d(
        x=Xf[mask],
        y=Yf[mask],
        z=Zf[mask],
        mode='markers',
        marker=dict(
            size=4,
            color=Df[mask],
            colorscale='Reds',
            opacity=0.8,
            colorbar=dict(title="Dust")
        )
    ))
    
    fig.update_layout(
        title="Dust Concentration Heatmap",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        height=600
    )
    
    return fig