
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from skeleton_functions import *




def visualize_compare_skeleton(skeleton1, skeleton2):
    rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0], [-sin(radians(-10)), 0, cos(radians(-10))]])
    coords1 = np.dot(skeleton1["joint_coordinates"], rotation_matrix)
    edges1 = skeleton1["edges"]
    coords2 = np.dot(skeleton2["joint_coordinates"], rotation_matrix)
    edges2 = skeleton2["edges"]

    plt.switch_backend('TkAgg')
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 5))
    pose_ax = fig.add_subplot(1, 2, 1, projection='3d')
    pose_ax.set_title('Prediction')
    range_ = np.amax(coords1)
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-range_, range_)
    plt.ylabel("y")
    plt.xlabel("x")
    for i_start, i_end in edges1:
        pose_ax.plot(*zip(coords1[i_start], coords1[i_end]), marker='o', markersize=2)

    pose_ax.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], s=2)


    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.set_title('Prediction')
    range_ = np.amax(coords2)
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-range_, range_)

    for i_start, i_end in edges2:
        pose_ax.plot(*zip(coords2[i_start], coords2[i_end]), marker='o', markersize=2)

    pose_ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], s=2)

    fig.tight_layout()
    plt.show()


def visualize_skeleton_from_file(skeleton_path, visualize_image = True):
    rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0], [-sin(radians(-10)), 0, cos(radians(-10))]])
    with open(skeleton_path, "rb") as skeleton_file:
        skeleton = pkl.load(skeleton_file)
        coords = np.dot(skeleton["joint_coordinates"], rotation_matrix)
        #coords = skeleton["joint_coordinates"]
        edges = skeleton['edges']
        #coords[:, 0], coords[:, 2] = coords[:, 2], -coords[:, 0]
        plt.switch_backend('TkAgg')
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(15, 15))
        if visualize_image:
            skeleton_path = skeleton_path.replace("NormalizedSkeletons", "PaddedFrames")
            skeleton_path = skeleton_path.replace("Skeletons", "PaddedFrames")
            skeleton_path = skeleton_path.replace("skeleton_", '')
            skeleton_path = skeleton_path.replace("normalized_", '')
            skeleton_path = skeleton_path.replace("pkl", "jpg")
            image = image_to_numpy(skeleton_path)[0]
            image_ax = fig.add_subplot(1, 2, 1)
            image_ax.set_title('Input')
            image_ax.imshow(image)

        pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
        pose_ax.set_title('Prediction')
        range_ = np.max(np.abs(coords))
        pose_ax.set_xlim3d(-range_, range_)
        pose_ax.set_ylim3d(-range_, range_)
        pose_ax.set_zlim3d(-range_, range_)
        plt.ylabel("y")
        plt.xlabel("x")
        for i_start, i_end in edges:
            pose_ax.plot(*zip(coords[i_start], coords[i_end]), marker='o', markersize=2)

        pose_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)
        fig.tight_layout()
        plt.show()



def visualize_skeleton(skeleton):
    rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0], [-sin(radians(-10)), 0, cos(radians(-10))]])
    coords = np.dot(skeleton["joint_coordinates"], rotation_matrix)
    edges = skeleton['edges']
    plt.switch_backend('TkAgg')
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    # Matplotlib interprets the Z axis as vertical, but our pose
    # has Y as the vertical axis.
    # Therefore we do a 90 degree rotation around the horizontal (X) axis
    #coords2 = coords.copy()
    #coords[:, 1], coords[:, 2] = coords2[:, 2], -coords2[:, 1]

    fig = plt.figure(figsize=(10, 5))

    pose_ax = fig.add_subplot(1, 1, 1, projection='3d')
    pose_ax.set_title('Prediction')
    range_ = np.amax(np.abs(skeleton["joint_coordinates"]))
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-range_, range_)
    plt.ylabel("y")
    plt.xlabel("x")
    for i_start, i_end in edges:
        pose_ax.plot(*zip(coords[i_start], coords[i_end]), marker='o', markersize=2)

    pose_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)

    fig.tight_layout()
    plt.draw()


def visualize_dtw(ex1, distance_measure):
    dist, path, sequence_dist = distance_measure(ex1)
    ex2 = ex1[:ex1.find("_")] + "_0"
    count = 0
    for idx in path:
        print("Skeleton #"+str(idx*5))
        print("Distance: "+str(sequence_dist[count]))
        count += 1

        sk1 = open_skeleton("Dataset/NormalizedSkeletons/"+ ex1 + "/normalized_skeleton_frame"+str(idx[0] * 5)+".pkl")
        sk2 = open_skeleton("Dataset/NormalizedSkeletons/"+ ex2 + "/normalized_skeleton_frame" + str(idx[1] * 5) + ".pkl")
        visualize_compare_skeleton(sk1, sk2)

