import streamlit as st
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import io
from sklearn.utils import resample

# --- Core ICP and Data Processing Functions ---

def surface_to_pc(surface_data, scale_xy=1.125, scale_z=1.0):
    """
    Converts a 2D surface (height map) numpy array into a 3D point cloud.
    
    Args:
        surface_data (np.ndarray): 2D array of height values.
        scale_xy (float): Scaling factor for x and y coordinates (e.g., pixel size).
        scale_z (float): Scaling factor for z coordinate (height).

    Returns:
        np.ndarray: (N, 3) array of 3D points.
    """
    height, width = surface_data.shape
    surface_mask = ~np.isnan(surface_data)
    x = np.arange(width) * scale_xy
    y = np.arange(height) * scale_xy
    xx, yy = np.meshgrid(x, y)
    zz = surface_data * scale_z
    points = np.vstack([xx[surface_mask].ravel(), yy[surface_mask].ravel(), zz[surface_mask].ravel()]).T
    return points

def o3d_point_to_plane_icp(source_pcd, target_pcd, threshold, max_iterations):
    """
    Runs Open3D's point-to-plane ICP.

    Args:
        source_pcd (o3d.geometry.PointCloud): The source point cloud to be transformed.
        target_pcd (o3d.geometry.PointCloud): The target point cloud.
        threshold (float): Correspondence distance threshold.
        max_iterations (int): Maximum number of ICP iterations.

    Returns:
        tuple: (transformation_matrix, fitness, inlier_rmse)
    """
    # Estimate normals for the target point cloud if they don't exist
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=threshold*3))

    # Run ICP
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    return reg_p2l.transformation, reg_p2l.fitness, reg_p2l.inlier_rmse

def generate_demo_surface(shape=(600, 1000)):
    """Creates an interesting surface for demonstration purposes."""
    height, width = shape
    x = np.linspace(-10, 10, width)
    y = np.linspace(-10, 10, height)
    xx, yy = np.meshgrid(x, y)
    # A surface with clear features (a sine wave peak and a cosine wave trough)
    zz = xx**2 - yy**2 + xx*yy + 5*np.sin(xx) + 10*np.cos(yy) - xx - 300*np.exp(-0.1*((xx)**2 + (yy)**2))#0.3*np.sin(2*xx + 1.5*yy) + 0.5*np.cos(0.8*xx**2 - yy**3) + 0.1*xx*yy + 0.05*xx**3
    return zz

# --- Visualization Functions ---

def create_3d_plot(points_dict, title):
    """
    Creates an interactive 3D plot of multiple point clouds using Plotly.
    
    Args:
        points_dict (dict): A dictionary where keys are names and values are (N, 3) point cloud arrays.
        title (str): The title of the plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()
    for i, (name, points) in enumerate(points_dict.items()):
        # Downsample for faster plotting if the cloud is very large
        if points.shape[0] > 10_000:
            points = resample(points, n_samples=10_000)

        color_defaults = ['blue', 'red', 'green']
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=1, color=color_defaults[i % len(color_defaults)]),
            name=name
        ))
    fig.update_layout(
        title_text=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(aspectmode='data'),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
    )
    return fig

# --- Streamlit UI Application ---

st.set_page_config(layout="wide", page_title="ICP Alignment Tool")

st.title("Interactive 3D ICP Alignment")
st.write("This tool allows you to align 3D point clouds derived from surface height maps using the Iterative Closest Point (ICP) algorithm. Use the sidebar to configure the settings.")

# --- Sidebar for Controls ---

with st.sidebar:
    st.header("Configuration")
    
    # 1. Mode Selection
    mode = st.radio(
        "Choose Operating Mode",
        ("1. Align Two Uploaded Surfaces", "2. Test with a Single Surface (Synthetic Transform)"),
        help="Choose 'Mode 1' to align two different scans. Choose 'Mode 2' to test the ICP algorithm's accuracy by applying a known transformation to a single scan and trying to recover it."
    )
    
    st.markdown("---")
    
    # 2. File Upload and Data Source
    source_surface, target_surface = None, None
    if mode == "1. Align Two Uploaded Surfaces":
        st.subheader("Upload Data")
        uploaded_target = st.file_uploader("Upload Target Surface CSV", type="csv")
        uploaded_source = st.file_uploader("Upload Source Surface CSV", type="csv")
        if uploaded_source and uploaded_target:
            source_surface = pd.read_csv(uploaded_source, header=None, skiprows=1).values
            target_surface = pd.read_csv(uploaded_target, header=None, skiprows=1).values
    else: # Mode 2
        st.subheader("Upload Data")
        uploaded_single = st.file_uploader("Upload a Single Surface CSV", type="csv")
        use_demo_data = st.checkbox("Or use generated demo data", value=not uploaded_single)
        if use_demo_data:
            st.success("Using a generated surface with distinct features for demonstration.")
            source_surface = generate_demo_surface()
        elif uploaded_single:
            source_surface = pd.read_csv(uploaded_single, header=None, skiprows=1).values

    st.markdown("---")

    # 3. Pre-processing and Voxelization
    st.subheader("Point Cloud Settings")
    scale_xy = 1.125
    scale_z = 1.
    st.markdown("---")

    # 4. Synthetic Transformation (Only for Mode 2)
    if mode == "2. Test with a Single Surface (Synthetic Transform)":
        st.subheader("Synthetic Transformation")
        st.write("Apply a known transformation to the 'source' part of the surface.")
        rot_x = st.slider("Rotation X (°)", -45.0, 45.0, 5.0, 0.5)
        rot_y = st.slider("Rotation Y (°)", -45.0, 45.0, -5.0, 0.5)
        rot_z = st.slider("Rotation Z (°)", -45.0, 45.0, 10.0, 0.5)
        trans_x = st.slider("Translation X", -1000.0, 1000.0, -20.0, 10.0)
        trans_y = st.slider("Translation Y", -1000.0, 1000.0, 15.0, 10.0)
        trans_z = st.slider("Translation Z", -20.0, 20.0, 5.0, 0.5)
        true_R_mat = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True).as_matrix()
        true_t_vec = np.array([trans_x, trans_y, trans_z])
    if mode == "1. Align Two Uploaded Surfaces":
        st.subheader("Pre-alignment Transformation")
        st.write("Apply a transformation to the 'source' to roughly align them.")
        rot_x = st.slider("Rotation X (°)", -45.0, 45.0, .0, 0.5)
        rot_y = st.slider("Rotation Y (°)", -45.0, 45.0, .0, 0.5)
        rot_z = st.slider("Rotation Z (°)", -45.0, 45.0, .0, 0.5)
        trans_x = st.slider("Translation X", -1000.0, 1000.0, 200.0, 10.0)
        trans_y = st.slider("Translation Y", -1000.0, 1000.0, .0, 10.0)
        trans_z = st.slider("Translation Z", -20.0, 20.0, .0, 0.5)
        prior_R_mat = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True).as_matrix()
        prior_t_vec = np.array([trans_x, trans_y, trans_z])

    # 5. ICP Parameters
    st.subheader("ICP Algorithm Parameters")
    icp_threshold = st.slider("Correspondence Threshold", 1.125, 20.0, 1.125 * 4, 1.125,
                              help="Maximum distance between two points to be considered a correspondence. Should be larger than the voxel size.")
    icp_iterations = st.slider("Max Iterations", 10, 200, 50, 5)

# --- Main Application Area ---
# Create a placeholder for the plot
plot_placeholder = st.empty()

# Check if we have data to visualize
if (mode == "1. Align Two Uploaded Surfaces" and source_surface is not None and target_surface is not None) or \
   (mode == "2. Test with a Single Surface (Synthetic Transform)" and source_surface is not None):
    
    with st.spinner("Preparing visualization..."):
        # --- 1. Prepare Point Clouds ---
        if mode == "1. Align Two Uploaded Surfaces":
            pc_source_full = surface_to_pc(source_surface, scale_xy, scale_z)
            pc_target_full = surface_to_pc(target_surface, scale_xy, scale_z)
            pc_source_full = resample(pc_source_full, n_samples=10000)
            pc_target_full = resample(pc_target_full, n_samples=10000)
            pc_source_full = (prior_R_mat @ pc_source_full.T).T + prior_t_vec
        
        else: # Mode 2: Split the single surface
            width = source_surface.shape[1]
            
            target_surface_part = source_surface[:, :int(width * 0.6)]
            source_surface_part = source_surface[:, int(width * 0.4):]

            pc_target_full = surface_to_pc(target_surface_part, scale_xy, scale_z)
            
            # Apply theoretical translation to move source part next to target
            pc_source_raw = surface_to_pc(source_surface_part, scale_xy, scale_z)
            pc_target_full = resample(pc_target_full, n_samples=10000)
            pc_source_raw = resample(pc_source_raw, n_samples=10000)
            translation_to_join = np.array([int(width * 0.4) * scale_xy, 0, 0])
            pc_source_full = pc_source_raw + translation_to_join
            # Apply the synthetic transformation
            center = np.nanmean(pc_source_full, axis=0)
            pc_source_full = (true_R_mat @ (pc_source_full - center).T).T + center + true_t_vec

        # --- 2. Create visualization ---
        fig_before = create_3d_plot({
            "Target": pc_target_full,
            "Source (Unaligned)": pc_source_full,
        }, "Current Alignment Preview") 
        
        # Display the plot in the placeholder
        plot_placeholder.plotly_chart(fig_before, use_container_width=True)

# Only run ICP when the button is clicked
if st.button("▶️ Run Alignment", type="primary"):
    if (mode == "1. Align Two Uploaded Surfaces" and source_surface is not None and target_surface is not None) or \
       (mode == "2. Test with a Single Surface (Synthetic Transform)" and source_surface is not None):
        
        with st.spinner("Running ICP... This may take a moment."):
            # --- 1. Prepare Point Clouds ---
            if mode == "1. Align Two Uploaded Surfaces":
                pc_source_full = surface_to_pc(source_surface, scale_xy, scale_z)
                pc_target_full = surface_to_pc(target_surface, scale_xy, scale_z)
                pc_source_full = resample(pc_source_full, n_samples=10000)
                pc_target_full = resample(pc_target_full, n_samples=10000)
                pc_source_full = (prior_R_mat @ pc_source_full.T).T + prior_t_vec
            
            else: # Mode 2: Split the single surface
                width = source_surface.shape[1]
                
                target_surface_part = source_surface[:, :int(width * 0.6)]
                source_surface_part = source_surface[:, int(width * 0.4):]

                pc_target_full = surface_to_pc(target_surface_part, scale_xy, scale_z)
                
                # Apply theoretical translation to move source part next to target
                pc_source_raw = surface_to_pc(source_surface_part, scale_xy, scale_z)
                pc_target_full = resample(pc_target_full, n_samples=10000)
                pc_source_raw = resample(pc_source_raw, n_samples=10000)
                translation_to_join = np.array([int(width * 0.4) * scale_xy, 0, 0])
                pc_source_full = pc_source_raw + translation_to_join
                # Apply the synthetic transformation
                center = np.nanmean(pc_source_full, axis=0)
                pc_source_full = (true_R_mat @ (pc_source_full - center).T).T + center + true_t_vec

            # --- 2. Create Open3D objects and downsample ---
            pcd_source = o3d.geometry.PointCloud()
            pcd_source.points = o3d.utility.Vector3dVector(pc_source_full)
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(pc_target_full)

            # --- 3. Visualization Before Alignment ---
            fig_before = create_3d_plot({
                "Target": pc_target_full,
                "Source (Unaligned)": pc_source_full,
            }, "Before Alignment") 

            # --- 4. Run ICP ---
            transform_mat, fitness, rmse = o3d_point_to_plane_icp(pcd_source, pcd_target, icp_threshold, icp_iterations)
            
            # --- 5. Apply Transformation ---
            pcd_source_aligned = o3d.geometry.PointCloud()
            pcd_source_aligned.points = pcd_source.points # Use full resolution for final visualization
            pcd_source_aligned.transform(transform_mat)

            # --- 6. Visualization After Alignment ---
            fig_after = create_3d_plot({
                "Target": np.asarray(pcd_target.points),
                "Source (Aligned)": np.asarray(pcd_source_aligned.points)
            }, "After Alignment")

            # --- 7. Display Results ---
            st.success("Alignment Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_before, use_container_width=True)
            with col2:
                st.plotly_chart(fig_after, use_container_width=True)

            st.subheader("Alignment Results")
            
            # Extract rotation and translation from the transformation matrix
            icp_R = transform_mat[:3, :3]
            icp_t = transform_mat[:3, 3]
            

            if mode == "2. Test with a Single Surface (Synthetic Transform)":
                recovered_true_R = np.array(icp_R.T, copy=True)  # Force writable copy
                r_angles_recovered = Rotation.from_matrix(recovered_true_R).as_euler('xyz', degrees=True)
                recovered_true_t = recovered_true_R @ center - center - recovered_true_R @ icp_t#- R_o3d_icp.T @ t_o3d_icp
                # In synthetic mode, we compare recovered vs true values
                st.write("Comparing the **recovered** transformation with the **true** synthetic one.")
                
                # Create a DataFrame for nice table display
                comparison_data = {
                    "Parameter": ["Rotation X (°)", "Rotation Y (°)", "Rotation Z (°)", "Translation X", "Translation Y", "Translation Z"],
                    "True Value": [rot_x, rot_y, rot_z, true_t_vec[0], true_t_vec[1], true_t_vec[2]],
                    # Note: ICP recovers the transformation to align source TO target. 
                    # The recovered transform is the *inverse* of the applied one. We invert it back for comparison.
                    # R_applied = R_recovered.T
                    # t_applied = -R_recovered.T @ t_recovered
                    # However, displaying the direct ICP output is more intuitive.
                    "Recovered Value": [r_angles_recovered[0], r_angles_recovered[1], r_angles_recovered[2], recovered_true_t[0], recovered_true_t[1], recovered_true_t[2]]
                }
                df = pd.DataFrame(comparison_data)
                df["Error"] = abs(df["True Value"] - df["Recovered Value"])
                st.dataframe(df.style.format({
                    "True Value": "{:.3f}", "Recovered Value": "{:.3f}", "Error": "{:.3f}"
                }), use_container_width=True)

            else: # Mode 1: Just show the recovered transformation
                final_R = icp_R @ prior_R_mat
                final_R_angles = Rotation.from_matrix(final_R).as_euler('xyz', degrees=True)
                final_t = icp_R @ prior_t_vec + icp_t#- R_o3d_icp.T @ t_o3d_icp
                st.write("The following transformation was calculated to align the **source** cloud to the **target** cloud.")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.write("**Final rotation transformation (Euler Angles):**")
                    st.code(f"X: {final_R_angles[0]:.3f}°\nY: {final_R_angles[1]:.3f}°\nZ: {final_R_angles[2]:.3f}°")
                with res_col2:
                    st.write("**Final translation Vector:**")
                    st.code(f"X: {final_t[0]:.3f}\nY: {final_t[1]:.3f}\nZ: {final_t[2]:.3f}")

            st.write(f"**ICP Fitness Score:** `{fitness:.4f}` (Overlap estimate)")
            st.write(f"**Inlier RMSE:** `{rmse:.4f}` (Mean distance between corresponding points)")

    else:
        st.error("Please upload the required CSV file(s) in the sidebar to proceed.")
