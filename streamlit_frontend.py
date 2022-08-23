import streamlit as st
import data_preparation as data
from optimized_clustering import OptimizedClustering
import result_visualization
from sklearn.preprocessing import MinMaxScaler
import copy

#-------------------------- Streamlit Configuration --------------------------#
# layout wide = use full page instead of small central column
st.set_page_config(layout="wide", page_title="Clustering App", page_icon="📊")
st.title("Comparison of Clustering Algorithms")

#---------------------------- Clustering Results -----------------------------#
clustering = None
X = data.df.copy()
unstripped_X = data.df.copy()

#---------------------------------- Sidebar ----------------------------------#
# User inputs:
with st.sidebar:
    # Selection: Features
    st.write("# Configuration:")
    selected_features = st.multiselect(
        "Features to consider: ",
        data.features,
        default=data.default_features
    )

    # Selection: Gender (if a gender column is given, add the option to filter by gender)
    if "gender" in data.features:
        selected_genders = st.multiselect(
            "Genders to consider: ",
            list(set(X["gender"])),
            default=list(set(X["gender"]))
        )
        X = X.loc[X["gender"].isin(selected_genders)]
    elif "Gender" in data.features:
        selected_genders = st.multiselect(
            "Select genders to consider: ",
            list(set(X["Gender"])),
            default=list(set(X["Gender"]))
        )
        X = X.loc[X["Gender"].isin(selected_genders)]

    # Selection: Number of allowed missing features
    num_allowed_sparse_features = st.selectbox(
        "How many missing feature values are allowed?",
        ["max 50%"] + list(range(X[selected_features].shape[1])))
    if num_allowed_sparse_features == "max 50%":
        num_allowed_sparse_features = 0.5
    X = data.drop_sparse_rows(X[selected_features],
                              num_allowed_sparse_features)
    retained_features_mask = [idx in X.index for idx in unstripped_X.index]

    # Selection: Scale type of the data
    scale_type = st.selectbox("How should the data be scaled?",
                              ["Scale each feture between 0 and 1",
                               "No additional Feature Scaling"])
    if scale_type == "Scale each feature between 0 and 1":
        scale_type = "normalize"
    elif scale_type == "No additional Feature Scaling":
        scale_type = "no_scaling"

    # Selection: Feature Selection mode
    fs_mode = st.selectbox("Additional Feature selection mode:",
                           ["Sequential Feature Selection",
                            "No additional Feature Selection"])
    if fs_mode == "Sequential Feature Selection":
        fs_mode = "sequential"
    elif fs_mode == "No additional Feature Selection":
        fs_mode = "no_fs"

    # Selection: Desired range for number of clusters k
    min_num_clusters = 2
    max_num_clusters = X.shape[0] - 1
    if max_num_clusters <= 100:  # use slider
        k_clusters_range = st.slider(
            "Desired range for number of clusters: ",
            min_value=min_num_clusters,
            max_value=max_num_clusters,
            value=(min_num_clusters, max_num_clusters)
        )
    else:
        st.write("Desired range for number of clusters: ")
        k_clusters_range_start = st.number_input(
            label="Minimum (>= {})".format(min_num_clusters),
            min_value=min_num_clusters,
            max_value=max_num_clusters,
            value=min_num_clusters
        )
        k_clusters_range_end = st.number_input(
            label="Maximum (<= {})".format(max_num_clusters),
            min_value=min_num_clusters,
            max_value=max_num_clusters,
            value=max_num_clusters
        )
        k_clusters_range = (
            min(k_clusters_range_start, k_clusters_range_end),
            max(k_clusters_range_start, k_clusters_range_end)
        )

    # Selection: Dowload a result table as png
    download_result_table = st.checkbox("Download result table")

    # Button: Start Cluster Analysis
    sidebar_col1, sidebar_col2, sidebar_col3 = st.columns(3)
    if sidebar_col2.button("Start Clustering"):
        try:
            if scale_type == "normalize":
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)

            X = data.impute_X(X)
            clustering = copy.deepcopy(OptimizedClustering(
                X=X,
                k_clusters_range=k_clusters_range,
                feature_selection_mode=fs_mode))
            clustering.fit()
        except ValueError:
            st.warning(
                "Something went wrong. Make sure your selections are correct.")

#---------------------------------- Results ----------------------------------#
if clustering != None:  # if clustering was applied
    col1, col2 = st.columns(2)

    # Plot SIL scores
    fig_scores = result_visualization.plot_silhouette_scores(
        clustering.result_infos)
    col1.write("### Silhouette Scores: ")
    col1.pyplot(fig=fig_scores, clear_figure=False)
    col1.write(
        "A score near -1 indicates a bad clustering (low separation & compactness) while a score near 1 indicates a good clustering (high separation & compactness).")

    # Plot cluster sizes & std.
    fig_sizes = result_visualization.plot_cluster_sizes_with_std(
        clustering.result_infos)
    col2.write("### Cluster Sizes & Standard Deviations: ")
    col2.pyplot(fig=fig_sizes, clear_figure=False)
    col2.write(
        "The colors stand for different clusters. Std. stands for the standard deviation of the cluster sizes. The lower the std., the more balanced are the cluster sizes (std=0 if all clusters have the same size).")

    # st.markdown("""---""")
    # Plot cluster assignments
    # fig_assigments = result_visualization.plot_cluster_assignments(
    #     clustering.result_infos)
    # st.write("### Cluster Assigments: ")
    # st.pyplot(fig=fig_assigments, clear_figure=True)
    # st.write("The colors show the cluster assignment of a data point.")

    # List detailed results
    result_visualization.print_all_results(
        clustering.result_infos, selected_features, retained_features_mask)

    # Dowload a result table as png if selected
    if download_result_table:
        png_name =\
            str(fs_mode) + "_" + \
            str(k_clusters_range) + "_" + \
            str(num_allowed_sparse_features) + "_" + \
            str(scale_type) + "_" + \
            str(selected_genders) + ".png"
        result_visualization.export_result_table_as_png(
            clustering.result_infos, png_name)
