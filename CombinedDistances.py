import os

import numpy as np

from skeleton_functions import *
from EuclideanDistances import euclidean_identify_repetitions


# Converte coordinate cartesiane in coordinate sferiche
def carth_to_sphere(x, y, z):
    r, theta, phi = cs.cart2sp(x=x, y=y, z=z)
    return np.array([r, phi, theta])


# Aggiunge coordinate sferiche a skeleton
def skeleton_to_sphere(skeleton):
    skeleton["sphere_coordinates"] = list([])
    for i in range(len(skeleton["joint_coordinates"])):
        joint = skeleton["joint_coordinates"][i]
        skeleton["sphere_coordinates"].append(carth_to_sphere(joint[0], joint[1], joint[2]))
    skeleton["sphere_coordinates"] = np.array(skeleton["sphere_coordinates"])
    return skeleton


# Distanza tra due punti definita autonomamente
def spheric_distance(A, B):
    rA = A[0]
    rB = B[0]
    phiA = A[1]
    phiB = B[1]
    thetaA = A[2]
    thetaB = B[2]
    if phiA == phiB and thetaA == thetaB:
        diff_an = 0
    else:
        diff_an = acos(cos(phiA - phiB) * cos(thetaA) * cos(thetaB) + sin(thetaA) * sin(thetaB))
    diff_r = np.linalg.norm(rA - rB)
    # [abs(rA-rB), abs(phiA-phiB), abs(tethaA-tethaB)]
    return diff_r + diff_an


# Distanza tra due skeleton
def spheric_skeleton_distance(skeletonA, skeletonB):
    diff = []
    for i in range(skeletonA.shape[0]):
        diff.append(spheric_distance(skeletonA[i], skeletonB[i]))
    return np.linalg.norm(diff)


def retrieve_sphere_PoI_sequences(exercise):
    PoI = euclidean_identify_repetitions(exercise)
    sequences = []
    index_list = []
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                skn = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl")
                skn = skeleton_to_sphere(skn)
                sequence.append(upper_body(skn, sphere=True))
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    elif exercise[:exercise.find("_")] in ["double-lunges", "single-lunges", "squat45"]:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                skn = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl")
                skn = skeleton_to_sphere(skn)
                sequence.append(lower_body(skn, sphere=True))
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    else:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                skn = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl")
                skn = skeleton_to_sphere(skn)
                sequence.append(skn["sphere_coordinates"])
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    return sequences, index_list


def sequence_sphere_distance(S1, S2):
    sequence_dist = []
    dist, path = fastdtw(S1, S2, dist=spheric_skeleton_distance)
    for idx in path:
        sequence_dist.append(spheric_skeleton_distance(S1[idx[0]], S2[idx[1]]))
    return dist / len(path), path, sequence_dist


def repetitions_combined_distance(exercise):
    user_sequences, user_index = retrieve_sphere_PoI_sequences(exercise)
    trainer_sequences, trainer_index = retrieve_sphere_PoI_sequences(exercise.replace(exercise[exercise.find("_"):], "_0"))
    distances = []
    i = 0
    if len(user_sequences) > len(trainer_sequences):
        while i < len(trainer_sequences):
            distances.append((i, i, sequence_sphere_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(user_sequences):
            distances.append((0, i, sequence_sphere_distance(trainer_sequences[0], user_sequences[i])))
            i += 1
    else:
        while i < len(user_sequences):
            distances.append((i, i, sequence_sphere_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(trainer_sequences):
            distances.append((i, 0, sequence_sphere_distance(trainer_sequences[i], user_sequences[0])))
            i += 1
    return distances, trainer_index, user_index


def retrieve_sphere_sequence(directory, body_part="full"):
    sequence = []
    for frame in os_sorted(os.listdir(directory)):
        with open(directory + "/" + frame, "rb") as skeleton_file:
            skeleton = pkl.load(skeleton_file)
            skeleton_to_sphere(skeleton)
            if body_part == "upper":
                sequence.append(upper_body(skeleton, sphere=True))
            elif body_part == "lower":
                sequence.append(lower_body(skeleton, sphere=True))
            else:
                sequence.append(skeleton["sphere_coordinates"])
    return sequence


def identify_combined_errors(exercise, repetition_distance, joint_thr_multiplier=1.0, frame_thr_multiplier=1.0, visualize_errors_flag=True):
    frames_number = len(os.listdir("Dataset/NormalizedSkeletons/"+exercise))
    joints_number = 12
    error_frame_list, repetition_error_list = identify_frame_errors(exercise, repetition_distance, frame_thr_multiplier)
    if len(error_frame_list)==0:
        return
    joint_error_counter = np.zeros(shape=(np.max(repetition_error_list)+1, open_normalized_skeleton(os.listdir("Dataset/NormalizedSkeletons")[0] + "/normalized_skeleton_frame0.pkl")["joint_coordinates"].shape[0]))
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        for j in range(len(error_frame_list)):
            frame_couple = error_frame_list[j]
            user_image = "Dataset/PaddedFrames/" + exercise + "/frame" + str(frame_couple[1]) + ".jpg"
            user_sk = skeleton_to_sphere(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl"))
            user_coordinates = upper_body(user_sk, sphere=True)
            trainer_sk = skeleton_to_sphere(open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl"))
            trainer_coordinates = upper_body(trainer_sk, sphere=True)
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_coordinates)):
                joint_distances.append((spheric_distance(user_coordinates[i], trainer_coordinates[i]), i))  # memorizza distanza e indice coordinata
            thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
            top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
            error_points = []
            for tuple in top_tier_distances:
                if tuple[0] > thr:
                    coords_idx = np.where(upper_body_names(user_sk)[tuple[1]] == user_sk["joint_names"])
                    error_points.append(tuple[1])
                    joint_error_counter[repetition_error_list[j]][coords_idx] += 1
            if visualize_errors_flag:
                visualize_errors(trainer_sk, user_sk, trainer_image, user_image, upper_body(user_sk)[error_points], upper_body(user_sk, two_dim=True)[error_points], frame_couple)

    elif exercise[:exercise.find("_")] in ["double-lunges", "single-lunges", "squat45"]:
        for j in range(len(error_frame_list)):
            frame_couple = error_frame_list[j]
            user_image = "Dataset/PaddedFrames/" + exercise + "/frame" + str(frame_couple[1]) + ".jpg"
            user_sk = skeleton_to_sphere(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl"))
            user_coordinates = lower_body(user_sk, sphere=True)
            trainer_sk = skeleton_to_sphere(open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl"))
            trainer_coordinates = lower_body(trainer_sk, sphere=True)
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_coordinates)):
                joint_distances.append((spheric_distance(user_coordinates[i], trainer_coordinates[i]), i))
            thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
            top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
            error_points = []
            for tuple in top_tier_distances:
                if tuple[0] > thr:
                    coords_idx = np.where(lower_body_names(user_sk)[tuple[1]] == user_sk["joint_names"])
                    error_points.append(tuple[1])
                    joint_error_counter[repetition_error_list[j]][coords_idx] += 1
            if visualize_errors_flag:
                visualize_errors(trainer_sk, user_sk, trainer_image, user_image, lower_body(user_sk)[error_points], lower_body(user_sk, two_dim=True)[error_points], frame_couple)
    else:
        joints_number=24
        for j in range(len(error_frame_list)):
            frame_couple = error_frame_list[j]
            user_image = "Dataset/PaddedFrames/" + exercise + "/frame" + str(frame_couple[1]) + ".jpg"
            user_sk = skeleton_to_sphere(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl"))
            user_coordinates = user_sk["sphere_coordinates"]
            trainer_sk = skeleton_to_sphere(open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl"))
            trainer_coordinates = trainer_sk["sphere_coordinates"]
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_coordinates)):
                joint_distances.append((spheric_distance(user_coordinates[i], trainer_coordinates[i]), i))
            thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
            top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
            error_points = []
            for tuple in top_tier_distances:
                if tuple[0] > thr:
                    coords_idx = np.where(user_sk["joint_names"][tuple[1]] == user_sk["joint_names"])
                    error_points.append(tuple[1])
                    joint_error_counter[repetition_error_list[j]][coords_idx] += 1
            if visualize_errors_flag:
                visualize_errors(trainer_sk, user_sk, trainer_image, user_image, user_sk["joint_coordinates"][error_points], user_sk["joint_2d_coordinates"][error_points],  frame_couple)
    MCE = np.argmax(np.sum(joint_error_counter, axis=0))
    print("L'articolazione che è stata maggiormente sbagliata nel corso dell'esercizio " + exercise[:exercise.find("_")] + " è: " + str(user_sk["joint_names"][MCE]) + " (" + str(
        int(np.sum(joint_error_counter, axis=0)[MCE])) + ")\tSuccesso esercizio: "+str(round((1-(np.sum(joint_error_counter))/(frames_number * joints_number))*100, 2))+"%")
    for i in range(joint_error_counter.shape[0]):
        MCE = np.argmax(joint_error_counter[i])
        print("L'articolazione che è stata maggiormente sbagliata nel corso della ripetizione " + str(i) + " è: " + str(user_sk["joint_names"][MCE]) + " (" + str(
            int(joint_error_counter[i][MCE])) + ")\tSuccesso ripetizione: "+str(round((1-(np.sum(joint_error_counter[i]))/((frames_number/len(np.unique(repetition_error_list))) * joints_number))*100, 2))+"%")



