from cmath import nan
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import streamlit as st
import pandas as pd
import dataframe_image as dfi


def plot_cluster_assignments(result_infos):
    """
    Return a plot of the cluster assignments obtained from different clustering algorithms present in result_infos. For each algorithm, plot every data point in one row (in the original order) having a specific color based on the cluster assignment.
    """
    n_samples = max(
        [len(algo_infos["cluster_assignments"]) for algo_infos in result_infos.values()])
    max_k_clusters = max(
        [len(algo_infos["cluster_infos"]) for algo_infos in result_infos.values()])
    assignment_colors = sns.color_palette(None, max_k_clusters)
    algo_names = []
    fig, ax = plt.subplots()

    i = 0
    for algo_name, algo_infos in result_infos.items():
        if algo_infos["score"] != "No result":
            i += 1
            algo_names.append(algo_name)
            assignments = algo_infos["cluster_assignments"]
            colors = [assignment_colors[i] for i in assignments]
            x = list(range(1, n_samples+1))
            y = [i]*n_samples
            ax.scatter(
                x, y, color=colors)
    plt.xlabel('Element Number')
    plt.xticks(np.arange(0, n_samples+1, 10))
    plt.xlim(0, n_samples+1)
    plt.grid(axis="x")
    plt.ylabel('Algorithm')
    plt.yticks(np.arange(1, len(algo_names)+1, 1), algo_names)

    return fig


def plot_cluster_sizes_with_std(result_infos):
    """
    Return a plot of the cluster sizes obtained for each clustering algorithm. Show the cluster sizes as a stacked bar chart for an algorithm and also show the standard deviation for the cluster sizes of an algorithm.
    """
    cluster_sizes = []
    labels = []
    max_k_clusters = max(
        [len(algo_infos["cluster_infos"]) for algo_infos in result_infos.values()])
    assignment_colors = sns.color_palette(None, max_k_clusters)

    for algo_name, algo_infos in result_infos.items():
        if algo_infos["score"] == "No result":
            continue  # Skip algorithms wihout result
        cluster_infos = algo_infos["cluster_infos"]
        sizes = [size for size in cluster_infos.values()]
        sizes += ([0] * (max_k_clusters-len(sizes)))
        cluster_sizes.append(sizes)

        labels.append(
            algo_name
            + "\n (Std.: "
            + str(round(algo_infos["result_std"], 2))
            + ")"
        )

    max_len_sizes = max(len(sizes) for sizes in cluster_sizes)

    fig, ax = plt.subplots()
    plt.xlabel("Size")
    left = [0]*len(labels)
    for cluster_i in range(max_len_sizes):
        cluster_i_sizes = [sizes[cluster_i] for sizes in cluster_sizes]
        cluster_i_color = assignment_colors[cluster_i]
        bars = ax.barh(y=labels, width=cluster_i_sizes,
                       color=cluster_i_color, left=left)
        ax.bar_label(bars, labels=[
                     i if i != 0 else "" for i in cluster_i_sizes], label_type='center')
        left = [sum(x) for x in zip(left, cluster_i_sizes)]

    return fig


def plot_silhouette_scores(result_infos):
    """
    Return a plot of the Silhouette scores of each clustering algorithm result. Show the result by bars having the legth matching the score and also color coded so that green=good and red=bad.
    """
    scores = []
    labels = []

    for algo_name, algo_infos in result_infos.items():
        score = algo_infos["score"]
        try:
            scores.append(round(float(score), 4))
            labels.append(algo_name)
        except:  # Skip algorithms wihout result
            continue
    bar_lengths = [score+1 for score in scores]
    normalized_scores = [length/2 for length in bar_lengths]

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_xticklabels([])
    ax.grid(axis="x")

    # Add colorbar for the x axis (score)
    silhouette_cmap = plt.cm.get_cmap("RdYlGn")
    colors = silhouette_cmap(normalized_scores)
    sm = ScalarMappable(cmap=silhouette_cmap,
                        norm=plt.Normalize(-1, 1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, orientation="horizontal", cax=cax)
    cbar.set_label('Score')

    bars = ax.barh(y=labels, width=bar_lengths, left=-1, color=colors)
    ax.bar_label(bars, labels=scores, label_type='center')

    return fig


def print_all_results(result_infos, features, retained_features_mask):
    """
    Print information about all applied clustering algorithms and their corresponding results in a streamlit frontend.
    """

    st.markdown("""---""")
    st.write("### Detailed Result Information: ")

    for algo_name, algo_info in result_infos.items():
        st.write("#### {}: ".format(algo_name))
        if algo_info["cluster_assignments"][0] == "No result":
            st.write("The Algorithm could not find a result.")
        else:
            # Create list of assignments with x for dropped columns
            unstripped_X_cluster_assignments = []
            count = 0
            for idx, i in enumerate(retained_features_mask):
                if i:
                    unstripped_X_cluster_assignments.append(
                        algo_info["cluster_assignments"][idx - count])
                else:
                    count += 1
                    unstripped_X_cluster_assignments.append(nan)

            st.write("Silhouette score:   " +
                     str(round(algo_info["score"], 2)))
            st.write("{} feature(s) selected: {}".format(str(algo_info["n_selected_features"]), str(
                np.asarray(features)[algo_info["feature_selection"]])))
            cluster_sizes = " | ".join("{}".format(size)
                                       for size in algo_info["cluster_infos"].values())
            st.write("Found {} clusters with sizes: {}".format(
                str(algo_info["k_selected_clusters"]), cluster_sizes))
            st.write("Cluster size standard deviation:   " +
                     str(round(algo_info["result_std"], 2)))
            st.write("\n" + "Cluster assignments for filtered data:   ")
            st.write(str(algo_info["cluster_assignments"]))
            st.write("\n" + "Cluster assignments for unfiltered data:   ")
            st.write(str(unstripped_X_cluster_assignments))

        st.markdown("""---""")


def export_result_table_as_png(result_infos, png_name="mytable.png"):
    export_table = []
    export_table_columns = ["Algorithm",
                            "SIL Score",
                            "Cluster Size Std.",
                            "k",
                            "Cluster Sizes",
                            "#Features"]
    for algo_name, algo_info in result_infos.items():
        if algo_info["cluster_assignments"][0] == "No result":
            st.write("The Algorithm could not find a result.")
            export_table_row = [algo_name] + \
                ["-"]*(len(export_table_columns)-1)
        else:
            cluster_sizes = " | ".join("{}".format(size)
                                       for size in algo_info["cluster_infos"].values())
            export_table_row = [
                algo_name,
                str(round(algo_info["score"], 2)),
                str(round(algo_info["result_std"], 2)),
                str(algo_info["k_selected_clusters"]),
                cluster_sizes,
                str(algo_info["n_selected_features"])
            ]

        export_table.append(export_table_row)

    df = pd.DataFrame(export_table, columns=export_table_columns)
    styled_df = df.style.\
        hide(axis='index').\
        apply(bold_max, axis=0, subset=["SIL Score"]).\
        apply(bold_min, axis=0, subset=["Cluster Size Std."])
    dfi.export(styled_df, png_name)


def bold_max(subset):
    bold = "font-weight: bold;"
    return [bold if cell == subset.max() else "" for cell in subset]


def bold_min(subset):
    bold = "font-weight: bold;"
    return [bold if cell == subset.min() else "" for cell in subset]
