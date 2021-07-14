import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
from natsort import os_sorted
import gc
import tensorflow as tf
from math import *
from fastdtw import fastdtw
import skimage.data
import skimage.transform
import skimage.io
import pickle as pkl
from scipy.spatial.distance import euclidean
from ai import cs
import requests
import zipfile
import errno, stat, shutil

def download_models(all=False):
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("models/metrabs_multiperson_smpl_combined"):
        url = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_multiperson_smpl_combined.zip'
        r = requests.get(url, allow_redirects=True)
        open('models/metrabs_multiperson_smpl_combined.zip', 'wb').write(r.content)
        with zipfile.ZipFile('models/metrabs_multiperson_smpl_combined.zip', 'r') as zip_ref:
            zip_ref.extractall("models")
        os.remove("models/metrabs_multiperson_smpl_combined.zip")

    if all:
        if not os.path.exists("models/metrabs_multiperson_smpl"):
            url = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_multiperson_smpl.zip'
            r = requests.get(url, allow_redirects=True)
            open('models/metrabs_multiperson_smpl.zip', 'wb').write(r.content)
            with zipfile.ZipFile('models/metrabs_multiperson_smpl.zip', 'r') as zip_ref:
                zip_ref.extractall("models")
            os.remove("models/metrabs_multiperson_smpl.zip")

        if not os.path.exists("models/metrabs_singleperson_smpl"):
            url = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_singleperson_smpl.zip'
            r = requests.get(url, allow_redirects=True)
            open('models/metrabs_singleperson_smpl.zip', 'wb').write(r.content)
            with zipfile.ZipFile('models/metrabs_singleperson_smpl.zip', 'r') as zip_ref:
                zip_ref.extractall("models")
            os.remove("models/metrabs_singleperson_smpl.zip")
    print("Models download finished!")

def image_to_numpy(image_path, dimensions=(256, 256)):
    # Load the image from image_path
    image = skimage.io.imread(image_path)
    image_numpy = np.stack([skimage.transform.resize(image, dimensions)])
    return image_numpy


def generate_skeleton(path):
    image = tf.image.decode_jpeg(tf.io.read_file(path))
    model = tf.saved_model.load('./models/metrabs_multiperson_smpl_combined')
    detections, poses3d, poses2d = model.predict_single_image(image)
    skeleton = {}
    skeleton["joint_coordinates"] = poses3d.numpy()
    skeleton["joint_coordinates"] = skeleton["joint_coordinates"][0]
    skeleton["joint_2d_coordinates"] = poses2d.numpy()
    skeleton["joint_2d_coordinates"] = skeleton["joint_2d_coordinates"][0]
    skeleton["edges"] = model.joint_edges.numpy()
    skeleton["joint_names"] = model.joint_names.numpy()
    return skeleton


def generate_skeletons(exercise):
    os.makedirs("Dataset/Skeletons", exist_ok=True)
    if os.path.exists("Dataset/Skeletons/" + exercise):
        return
    os.makedirs("Dataset/Skeletons/" + exercise, exist_ok=True)
    images = tf.stack([tf.image.decode_jpeg(tf.io.read_file("./Dataset/PaddedFrames/" + exercise + "/frame0.jpg"))], axis=0)
    images_list = os_sorted(os.listdir("Dataset/PaddedFrames/" + exercise))
    for frame in range(5, len(images_list), 5):
        images = tf.concat([images, [tf.image.decode_jpeg(tf.io.read_file("./Dataset/PaddedFrames/" + exercise + "/" + images_list[frame]))]], axis=0)
    model = tf.saved_model.load("./models/metrabs_multiperson_smpl_combined")
    detections, poses3d, poses2d = model.predict_multi_image(images)

    for i in range(len(images)):
        if poses3d[i].numpy().size == 0:
            with open("Dataset/SkeletonsNonRiusciti.txt", "a+") as file:
                file.write(exercise + " - frame" + str(i * 5) + "\n")
        else:
            skeleton = {}
            skeleton["joint_coordinates"] = poses3d[i].numpy()
            skeleton["joint_coordinates"] = skeleton["joint_coordinates"][0]
            skeleton["joint_2d_coordinates"] = poses2d[i].numpy()
            skeleton["joint_2d_coordinates"] = skeleton["joint_2d_coordinates"][0]
            skeleton["edges"] = model.joint_edges.numpy()
            skeleton["joint_names"] = model.joint_names.numpy()
            with open("Dataset/Skeletons/" + exercise + "/skeleton_frame" + str(i * 5) + ".pkl", "wb") as skf:
                pkl.dump(skeleton, skf)


def generate_skeleton_dataset(Frames_path):
   # pad_frames(Frames_path)
    #for exercise in os_sorted(os.listdir("Dataset/PaddedFrames")):
     #   print("Generating skeletons of: " + exercise)
       # generate_skeletons(exercise)
    #skeleton_fixer()
    renamer()

def pad_frames(Frames_path):
    for directory in os_sorted(os.listdir(Frames_path)):
        print("Preparo la cartella:" + directory)
        os.system("python ImagesPreProcessing.py --directory= " + Frames_path + "/" + directory)


def skeleton_fixer():
    model = tf.saved_model.load('./models/metrabs_multiperson_smpl_combined')
    with open("Dataset/SkeletonsNonRiusciti.txt", "r") as file:
        for line in file:
            idx = line.find("frame")
            exercise = line[0:(idx - 3)]
            frame = int(line[idx + 5:len(line) - 1])
            if frame != 0 and frame < len(os.listdir("Dataset/PaddedFrames/" + exercise)) - 5:
                frames_pool = list(range(frame - 4, frame + 5))
                i = 1
                while i < int(len(frames_pool) / 2):
                    for sgn in [-1, 1]:
                        idx = int(len(frames_pool) / 2) + (i * sgn)
                        image = tf.image.decode_jpeg(tf.io.read_file("Dataset/PaddedFrames/" + exercise + "/frame" + str(frames_pool[idx]) + ".jpg"))
                        detections, poses3d, poses2d = model.predict_single_image(image)
                        if poses3d.numpy().size != 0:
                            skeleton = {}
                            skeleton["joint_coordinates"] = poses3d.numpy()
                            skeleton["joint_coordinates"] = skeleton["joint_coordinates"][0]
                            skeleton["joint_2d_coordinates"] = poses2d.numpy()
                            skeleton["joint_2d_coordinates"] = skeleton["joint_2d_coordinates"][0]
                            skeleton["edges"] = model.joint_edges.numpy()
                            skeleton["joint_names"] = model.joint_names.numpy()
                            print("############   Sostituito frame " + str(frame) + "con: " + str(frames_pool[idx]))
                            with open("Dataset/Skeletons/" + exercise + "/skeleton_frame" + str(frame) + ".pkl", "wb") as skf:
                                pkl.dump(skeleton, skf)
                            i = 1000000
                            break
                        else:
                            i += 1

        with open("Dataset/SkeletonsNonRiusciti.txt", "w") as file:
            for exercise in os_sorted(os.listdir("Dataset/Skeletons")):
                frames = os_sorted(os.listdir("Dataset/Skeletons/" + exercise))
                for idx in range(0, len(frames), 5):
                    frame = "skeleton_frame" + str(idx) + ".pkl"
                    if frame not in frames:
                        file.write(exercise + " - frame" + str(idx) + "\n")



def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise


def renamer():
    os.makedirs("Dataset/SkeletonsNew", exist_ok=True)
    for exercise in os_sorted(os.listdir("Dataset/Skeletons")):
        os.makedirs("Dataset/SkeletonsNew/" + exercise, exist_ok=True)
        frames = os_sorted(os.listdir("Dataset/Skeletons/" + exercise))
        i = 0
        for frame in frames:
            with open("Dataset/Skeletons/" + exercise + "/" + frame, "rb") as f:
                skeleton = pkl.load(f)
            with open("Dataset/SkeletonsNew/" + exercise + "/skeleton_frame" + str(i) + ".pkl", "wb") as f:
                pkl.dump(skeleton, f)
            i += 5
    shutil.rmtree("Dataset/Skeletons", ignore_errors=False, onerror=handleRemoveReadonly)
    os.rename("Dataset/SkeletonsNew", "Dataset/Skeletons")


def upper_body(skeleton, sphere=False, two_dim=False):
    coords = list(range(12, 24))
    if sphere:
        return skeleton["sphere_coordinates"][coords]
    if two_dim:
        return skeleton["joint_2d_coordinates"][coords]
    return skeleton["joint_coordinates"][coords]


def upper_body_names(skeleton):
    coords = list(range(12, 24))
    return skeleton["joint_names"][coords]


def lower_body(skeleton, sphere=False, two_dim=False):
    coords = list(range(12))
    if sphere:
        return skeleton["sphere_coordinates"][coords]
    if two_dim:
        return skeleton["joint_2d_coordinates"][coords]
    return skeleton["joint_coordinates"][coords]


def lower_body_names(skeleton):
    coords = list(range(12))
    return skeleton["joint_names"][coords]


def find_coord(skeleton, name):
    return skeleton["joint_coordinates"][np.where(name == skeleton["joint_names"])]


def find_2d_coord(skeleton, name):
    return skeleton["joint_2d_coordinates"][np.where(name == skeleton["joint_names"])]


def normalize_skeleton(skeleton_path, autosave=True, delete_old=False):
    with open(skeleton_path, "rb") as skeleton_file:
        skeleton = pkl.load(skeleton_file)

    center_points = [b'pelv', b'rhip', b'lhip']
    # Estrazione vettore medio fianchi e torso
    mean_vector = np.zeros(dtype='float64', shape=(1, 3))
    for joint_name in center_points:
        index = np.where(skeleton["joint_names"] == joint_name)
        mean_vector += skeleton["joint_coordinates"][index]
    mean_vector = mean_vector / len(center_points)
    mean_vector = np.reshape(mean_vector, (3,))

    #  Normalizzazione vettore dei joints (centramento sul baricentro e normalizzazione)
    for joint in skeleton["joint_coordinates"]:
        joint -= mean_vector
    skeleton["joint_coordinates"] = skeleton["joint_coordinates"] / np.linalg.norm(skeleton["joint_coordinates"])

    # Calcolo matrice M per la normalizzazione dell'inquadratura
    Jhl = np.transpose(skeleton["joint_coordinates"][np.where(skeleton["joint_names"] == b'lcla')])
    Jt = np.transpose(skeleton["joint_coordinates"][np.where(skeleton["joint_names"] == b'rcla')])
    Jhl = Jhl / np.linalg.norm(Jhl)
    norm2 = np.linalg.norm(Jhl)
    tras = np.transpose(Jhl)
    temp = np.dot((tras / norm2), Jt).item()
    Jhl_ort = Jt - (temp * Jhl)

    Jhl_ort = Jhl_ort / np.linalg.norm(Jhl_ort)
    cross_product_vector = np.cross(Jhl, Jhl_ort, axis=0)

    cross_product_vector = cross_product_vector / np.linalg.norm(cross_product_vector)
    M = np.concatenate((Jhl, Jhl_ort, cross_product_vector), axis=1)
    x_tilde = np.transpose(skeleton["joint_coordinates"])
    x_tilde = np.dot(np.transpose(M), x_tilde)
    skeleton["joint_coordinates"] = np.transpose(x_tilde)

    if autosave:
        if os.path.exists(skeleton_path) and delete_old:
            os.remove(skeleton_path)
        skeleton_path = skeleton_path.replace("skeleton", "normalized_skeleton")
        skeleton_path = skeleton_path.replace("Skeletons", "NormalizedSkeletons")
        with open(skeleton_path, "wb") as skeleton_file:
            pkl.dump(skeleton, skeleton_file)
    return skeleton


def is_ort(P1, P2):
    if np.inner(np.transpose(P1), np.transpose(P2)) <= 0.00001:
        return True
    else:
        return False


def normalize_dataset():
    os.makedirs("Dataset/NormalizedSkeletons", exist_ok=True)
    for directory in os_sorted(os.listdir("Dataset/Skeletons")):
        os.makedirs("Dataset/NormalizedSkeletons/" + directory, exist_ok=True)
        for frame in os_sorted(os.listdir("Dataset/Skeletons/" + directory)):
            normalize_skeleton("Dataset/Skeletons/" + directory + "/" + frame, autosave=True, delete_old=False)


def normalize_directory(directory):
    os.makedirs("Dataset/NormalizedSkeletons/" + directory, exist_ok=True)
    for frame in os_sorted(("Dataset/Skeletons/" + directory)):
        normalize_skeleton("Dataset/Skeletons/" + directory + "/" + frame, autosave=True, delete_old=False)


def skeleton_regularization(exercise, SW_size=5, autosave=False):
    weights = np.ones(shape=(SW_size - 1,))
    for i in range(int(len(weights) / 2)):
        weights[i] = 1 / (2 ** (int(len(weights) / 2) - i))
    for i in range(int(len(weights) / 2), len(weights)):
        weights[i] = 1 / (2 ** (i - int((len(weights) / 2) - 1)))
    sliding_window = []
    copy = []
    for i in range(SW_size):
        sk = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(i * 5) + ".pkl")
        sliding_window.append(sk)
        copy.append(sk["joint_coordinates"])

    for idx in range(int(SW_size / 2), len(os.listdir("Dataset/NormalizedSkeletons/" + exercise)) - int(SW_size / 2)):
        copy = []
        for i in range(SW_size):
            if i != int(SW_size / 2) + 1:
                copy.append(sliding_window[i]["joint_coordinates"])
        copy = [copy[i] * weights[i] for i in range(len(weights))]

        for joint in range(sliding_window[0]["joint_coordinates"].shape[0]):
            copy = np.array(copy)
            mean_joint = copy[0:, joint]
            mean = sum(mean_joint) / sum(weights)
            diff = np.absolute(mean - (sliding_window[int(SW_size / 2)]["joint_coordinates"][joint]))
            tollerance = np.absolute(mean * 2)
            if (diff > tollerance).any():
                # visualize_skeleton(sliding_window[int(SW_size/2)])
                sliding_window[int(SW_size / 2)]["joint_coordinates"][joint] = mean
                # visualize_skeleton(sliding_window[int(SW_size / 2)])
                if autosave:
                    with open("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame" + str(idx * 5) + ".pkl", "wb") as skf:
                        pkl.dump(sliding_window[int(SW_size / 2)], skf)
        sliding_window.pop(0)
        if idx + 3 >= len(os.listdir("Dataset/NormalizedSkeletons/" + exercise)):
            return
        sliding_window.append(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str((idx + 3) * 5) + ".pkl"))


def visualize_skeleton(skeleton):
    coords = skeleton["joint_coordinates"]
    edges = skeleton['edges']

    plt.switch_backend('TkAgg')
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    # Matplotlib interprets the Z axis as vertical, but our pose
    # has Y as the vertical axis.
    # Therefore we do a 90 degree rotation around the horizontal (X) axis
    # coords2 = coords.copy()
    # coords[:, 1], coords[:, 2] = coords2[:, 2], -coords2[:, 1]

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
    plt.show()


def open_skeleton(skeleton_path):
    with open(skeleton_path, "rb") as skeleton_file:
        sk = pkl.load(skeleton_file)
    return sk


def open_normalized_skeleton(skeleton_path):
    with open("Dataset/NormalizedSkeletons/" + skeleton_path, "rb") as skeleton_file:
        sk = pkl.load(skeleton_file)
    return sk




def retrieve_sequence(directory, body_part="full"):
    sequence = []
    for frame in os_sorted((directory)):
        with open(directory + "/" + frame, "rb") as skeleton_file:
            skeleton = pkl.load(skeleton_file)
            if body_part == "upper":
                sequence.append(upper_body(skeleton))
            elif body_part == "lower":
                sequence.append(lower_body(skeleton))
            else:
                sequence.append(skeleton["joint_coordinates"])
    return sequence


def identify_frame_errors(exercise, repetition_distance, thr_multiplier=1.0):
    error_list = []
    repetition_list = []
    distances, trainer_index, user_index = repetition_distance(exercise)
    for triple in distances:
        user_repetition_num = triple[1]
        trainer_repetition_num = triple[0]
        path = triple[2][1]
        skeleton_distances = triple[2][2]
        thr = np.mean(skeleton_distances) * thr_multiplier
        for i in range(len(skeleton_distances)):
            if skeleton_distances[i] > thr:
                user_frame = user_index[user_repetition_num][path[i][1]]
                trainer_frame = trainer_index[trainer_repetition_num][path[i][0]]
                error_list.append((trainer_frame, user_frame))
                repetition_list.append(user_repetition_num)
    print("Errori commessi: " + str(len(error_list)))
    print("Nelle coppie di frame: " + str(error_list))
    return error_list, repetition_list


def visualize_errors(trainer_sk, user_sk, trainer_image, user_image, error_points, error_2d_points, frame_couple):
    if len(error_points) == 0:
        print("Joint Errors not detected in frame " + str(frame_couple[1]) + "! (Try to set a lower threshold.)")
        return

    rotation_matrix = np.dot(np.dot([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), [[cos(radians(-10)), 0, sin(radians(-10))], [0, 1, 0], [-sin(radians(-10)), 0, cos(radians(-10))]])

    coords1 = np.dot(trainer_sk["joint_coordinates"], rotation_matrix)
    # coords1 = trainer_sk["joint_coordinates"]
    edges1 = trainer_sk["edges"]
    coords2 = np.dot(user_sk["joint_coordinates"], rotation_matrix)
    # coords2 = user_sk["joint_coordinates"]
    edges2 = user_sk["edges"]
    error_points = np.dot(np.array(error_points), rotation_matrix)
    # error_points = np.array(error_points)

    matplotlib.use('Qt5Agg')
    # noinspection PyUnresolvedReferences
    #from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 15))

    trainer_image = image_to_numpy(trainer_image)[0]
    image_ax = fig.add_subplot(2, 2, 1)
    image_ax.set_title('\nTRAINER')
    image_ax.imshow(trainer_image)

    user_image = image_to_numpy(user_image)[0]
    image_ax = fig.add_subplot(2, 2, 2)
    image_ax.set_title('\nUSER')
    color = [[1, 0, 0]]
    i = 1
    for error in error_2d_points:
        image_ax.scatter(error[0], error[1], marker='X', c=color, s=100)
        color[0][1] += 1 / (2 ** i)
        i += 1

    image_ax.imshow(user_image)

    pose_ax = fig.add_subplot(2, 2, 3, projection='3d')
    pose_ax.set_title('\nFrame:' + str(frame_couple[0]))
    range_ = np.amax(np.abs(coords1))
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-range_, range_)
    pose_ax.set_ylabel(ylabel='y')
    pose_ax.set_xlabel("x")
    pose_ax.set_zlabel("z")

    for i_start, i_end in edges1:
        pose_ax.plot(*zip(coords1[i_start], coords1[i_end]), marker='o', markersize=2)

    pose_ax.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], c='#0000ff', s=2)

    pose_ax = fig.add_subplot(2, 2, 4, projection='3d')
    pose_ax.set_title('\nFrame:' + str(frame_couple[1]))
    range_ = np.amax(np.abs(coords2))
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-range_, range_)
    pose_ax.set_ylabel(ylabel='y')
    pose_ax.set_xlabel("x")
    pose_ax.set_zlabel("z")

    for i_start, i_end in edges2:
        pose_ax.plot(*zip(coords2[i_start], coords2[i_end]), marker='o', markersize=2)

    pose_ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], c='#0000ff', s=2)
    color = [[1, 0, 0]]
    i = 1
    for error in error_points:
        pose_ax.scatter(error[0], error[1], error[2], marker='X', c=color, s=50)
        color[0][1] += 1 / (2 ** i)
        i += 1
    fig.tight_layout(pad=0.1, h_pad=0.01, w_pad=0.01, rect=(0, 0, 1, 1))

    plt.show()
    return



def main():
    parser = argparse.ArgumentParser(description='Joint Dataset Converter', allow_abbrev=False)
    parser.add_argument('--directory', type=str, required=True)
    parser.add_argument('--frame', type=str, required=True)
    opts = parser.parse_args()
    tf.disable_v2_behavior()

    os.makedirs(os.path.join(os.path.abspath(os.getcwd()), "Dataset\\Skeletons"), exist_ok=True)
    joint_extractor_from_directory(opts.directory, int(opts.frame))


if __name__ == "__main__":
    main()


################   FUNZIONI BASATE SU METROPOSE3D  (DEPRECATE)   ###################

def joint_generator(image_path, model="cocoapi/Model/many_rn50_st32.pb"):
    # 1. Get an image tensor. It could be a placeholder or an input pipeline using tf.data as well.
    images_tensor = tf.convert_to_tensor(image_to_numpy(image_path), dtype=tf.float32)

    # 2. Build the pose estimation graph from the exported model
    # That file also contains the joint names and skeleton edge connectivity as well.
    poses_tensor, edges_tensor, joint_names_tensor = estimate_pose(images_tensor, model)

    # 3. Variable definitions
    skeleton = {"edges": None, "joint_coordinates": None, "joint_names": None}

    # 4. Run the actual estimation
    with tf.Session() as sess:
        # matrice adiacenza degli archi che collegano i joints (18x2)
        skeleton["edges"] = sess.run(edges_tensor)

        # coordinate dei joint dello skeleton (3D)(19x3)
        skeleton["joint_coordinates"] = sess.run(poses_tensor)[0]  # joint_coordinates ha lo [0] perchè almeno viene rappresentato come una matrice (19x3)

        # Nomi di ogni joint, sono nell'ordine in cui compaiono in joint_coordinates (19x1)
        skeleton["joint_names"] = sess.run(joint_names_tensor)

    sess.close()
    return skeleton


def estimate_pose(images_tensor, model_path):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    poses_op, edges_op, joint_names_op = tf.import_graph_def(graph_def, input_map={'input:0': images_tensor}, return_elements=['output', 'joint_edges', 'joint_names'])
    poses_tensor = poses_op.outputs[0]
    edges_tensor = edges_op.outputs[0]
    joint_names_tensor = joint_names_op.outputs[0]
    return poses_tensor, edges_tensor, joint_names_tensor


# A causa dei leak di memoria è inutilizzabile (DEPRECATA)
def joint_extractor(frames_path="Dataset/Frames/"):
    os.makedirs(os.path.join(os.path.abspath(os.getcwd()), "Dataset\\Skeletons"), exist_ok=True)
    for directory in os_sorted((frames_path)):
        print("Sto calcolando gli skeleton di: " + directory)
        counter = 0
        os.makedirs(os.path.join(os.path.abspath(os.getcwd()), "Dataset\\Skeletons\\", str(directory)), exist_ok=True)
        for frame in os_sorted(os.listdir(frames_path + directory)):
            start = time.time()
            if counter % 5 == 0 or counter == 0:
                skeleton = joint_generator(os.path.join(frames_path, directory, frame))
                with open("Dataset\\Skeletons\\" + str(directory) + "\\skeleton" + "_frame" + str(counter) + ".pkl", "wb") as skeleton_file:
                    pkl.dump(skeleton, skeleton_file)
                print("Frame: " + str(counter) + "\tTime= " + str(time.time() - start))
            gc.collect()
            counter += 1


def joint_extractor_from_directory(extraction_dir, begin_frame=0, skeleton_output=15,
                                   extraction_dir_path="Dataset/PaddedFrames/"):  # skeleton_output indica il numero di skeleton che si vogliono ottenere
    print("Sto calcolando gli skeleton di: " + extraction_dir)
    for idx in range(begin_frame, begin_frame + (skeleton_output * 5)):
        start = time.time()
        if idx % 5 == 0 or idx == 0:
            skeleton = joint_generator(os.path.join(extraction_dir_path, extraction_dir, "frame" + str(idx) + ".jpg"))
            os.makedirs("Dataset/Skeletons/" + str(extraction_dir), exist_ok=True)
            with open("Dataset\\Skeletons\\" + str(extraction_dir) + "\\skeleton" + "_frame" + str(idx) + ".pkl", "wb") as skeleton_file:
                pkl.dump(skeleton, skeleton_file)
            print("Frame: " + str(idx) + "\tTime= " + str(time.time() - start))
