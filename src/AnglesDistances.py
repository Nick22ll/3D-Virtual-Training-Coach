
from skeleton_functions import *
from EuclideanDistances import euclidean_identify_repetitions


#####################     ANGLES FUNCTIONS       #######################

#  Calcola l'angolo in radianti tra i vettori u e v
def angle(skeleton, A, B, C):
    u = find_vector(find_coord(skeleton, A), find_coord(skeleton, B))
    v = find_vector(find_coord(skeleton, C), find_coord(skeleton, B))
    cos = np.inner(u, v).item() / (np.linalg.norm(u) * np.linalg.norm(v))
    return acos(cos)


def find_vector(P, Q):
    vect = P - Q
    norm = np.linalg.norm(vect)
    return vect / norm


def retrieve_angles(skeleton, body_part="full"):
    angles = []
    index = []
    if body_part == "upper" or body_part == "full":
        angles.append(angle(skeleton, b'neck', b'spin', b'pelv'))
        angles.append(angle(skeleton, b'thor', b'neck', b'lsho'))
        angles.append(angle(skeleton, b'lcla', b'lsho', b'lelb'))
        angles.append(angle(skeleton, b'lsho', b'lelb', b'lwri'))
        angles.append(angle(skeleton, b'thor', b'neck', b'rsho'))
        angles.append(angle(skeleton, b'rcla', b'rsho', b'relb'))
        angles.append(angle(skeleton, b'rsho', b'relb', b'rwri'))
        angles.append(angle(skeleton, b'head', b'neck', b'rcla'))
        angles.append(angle(skeleton, b'head', b'neck', b'lcla'))
        angles.append(angle(skeleton, b'lsho', b'lcla', b'neck'))
        angles.append(angle(skeleton, b'rsho', b'rcla', b'neck'))
        for joint in [b'spin',b'lsho', b'lsho', b'lelb', b'rsho', b'rsho', b'relb', b'neck', b'neck', b'lcla', b'rcla']:
            index.append(joint)

    elif body_part == "lower" or body_part == "full":
        angles.append(angle(skeleton, b'neck', b'pelv', b'lhip'))
        angles.append(angle(skeleton, b'pelv', b'lhip', b'lkne'))
        angles.append(angle(skeleton, b'lhip', b'lkne', b'lank'))
        angles.append(angle(skeleton, b'neck', b'pelv', b'rhip'))
        angles.append(angle(skeleton, b'pelv', b'rhip', b'rkne'))
        angles.append(angle(skeleton, b'rhip', b'rkne', b'rank'))
        angles.append(angle(skeleton, b'rkne', b'rank', b'rtoe'))
        angles.append(angle(skeleton, b'lkne', b'lank', b'ltoe'))
        angles.append(angle(skeleton, b'thor', b'spin', b'bell'))
        angles.append(angle(skeleton, b'spin', b'bell', b'pelv'))
        angles.append(angle(skeleton, b'bell', b'pelv', b'rhip'))
        angles.append(angle(skeleton, b'bell', b'pelv', b'lhip'))
        for joint in [b'lhip', b'lhip', b'lkne', b'rhip', b'rhip', b'rkne', b'rank', b'lank', b'spin', b'bell', b'pelv', b'pelv']:
            index.append(joint)

    angles = np.array(angles)
    return angles, index


def retrieve_angles_sequence(directory, body_part="full"):
    sequence = []
    for frame in os_sorted((directory)):
        with open(directory + "/" + frame, "rb") as skeleton_file:
            skeleton = pkl.load(skeleton_file)
            sequence.append(retrieve_angles(skeleton, body_part=body_part)[0])
    return sequence


def angles_distance(anglesA, anglesB):
    return np.linalg.norm(anglesA - anglesB)


def retrive_angles_PoI_sequences(exercise):
    PoI = euclidean_identify_repetitions(exercise)
    sequences = []
    index_list = []
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                sequence.append(retrieve_angles(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl"), body_part="upper")[0])
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    elif exercise[:exercise.find("_")] in ["double-lunges", "single-lunges", "squat45"]:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                sequence.append(retrieve_angles(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl"), body_part="lower")[0])
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    else:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                sequence.append(retrieve_angles(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl"))[0])
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    return sequences, index_list


def repetitions_angles_distance(exercise):
    user_sequences, user_index = retrive_angles_PoI_sequences(exercise)
    trainer_sequences, trainer_index = retrive_angles_PoI_sequences(exercise.replace(exercise[exercise.find("_"):], "_0"))
    distances = []
    i = 0
    if len(user_sequences) > len(trainer_sequences):
        while i < len(trainer_sequences):
            distances.append((i, i, sequence_angles_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(user_sequences):
            distances.append((0, i, sequence_angles_distance(trainer_sequences[0], user_sequences[i])))
            i += 1
    else:
        while i < len(user_sequences):
            distances.append((i, i, sequence_angles_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(trainer_sequences):
            distances.append((i, 0, sequence_angles_distance(trainer_sequences[i], user_sequences[0])))
            i += 1
    return distances, trainer_index, user_index


def sequence_angles_distance(S1, S2):
    sequence_dist = []
    dist, path = fastdtw(S1, S2, dist=angles_distance)
    for idx in path:
        sequence_dist.append(angles_distance(S1[idx[0]], S2[idx[1]]))
    return dist / len(path), path, sequence_dist


def identify_angles_errors(exercise, repetition_distance, joint_thr_multiplier=1.0, frame_thr_multiplier=1.0, visualize_errors_flag = True):
    frames_number = len(os.listdir("Dataset/NormalizedSkeletons/" + exercise))
    joints_number = 12
    error_frame_list, repetition_error_list = identify_frame_errors(exercise, repetition_distance, frame_thr_multiplier)
    if len(error_frame_list)==0:
        return
    joint_error_counter = np.zeros(shape=(np.max(repetition_error_list)+1, open_normalized_skeleton(os.listdir("Dataset/NormalizedSkeletons")[0] + "/normalized_skeleton_frame0.pkl")["joint_coordinates"].shape[0]))
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        for j in range(len(error_frame_list)):
            frame_couple = error_frame_list[j]
            user_image = "Dataset/PaddedFrames/" + exercise + "/frame" + str(frame_couple[1]) + ".jpg"
            user_sk = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl")
            user_angles, user_angles_index = retrieve_angles(user_sk, body_part="upper")
            trainer_sk = open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl")
            trainer_angles, trainer_angles_index = retrieve_angles(trainer_sk, body_part="upper")
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_angles)):
                joint_distances.append((angles_distance(user_angles[i], trainer_angles[i]), i))  # memorizza distanza e indice coordinata
            thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
            top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
            error_points = []
            for tuple in top_tier_distances:
                if tuple[0] > thr:
                    coords_idx = np.where(upper_body_names(user_sk)[tuple[1]] == user_sk["joint_names"])
                    error_points.append(tuple[1])
                    joint_error_counter[repetition_error_list[j]][coords_idx] += 1
            error_points = [user_angles_index[i] for i in error_points]
            errors = []
            errors_2d = []
            for idx in range(len(error_points)):
                errors.append(np.reshape(find_coord(user_sk, error_points[idx]), newshape=(3,)))
                errors_2d.append(np.reshape(find_2d_coord(user_sk, error_points[idx]), newshape=(3,)))
            errors = np.array(errors)
            if visualize_errors_flag:
                visualize_errors(trainer_sk, user_sk, trainer_image, user_image, errors, errors_2d, frame_couple)

    elif exercise[:exercise.find("_")] in ["double-lunges", "single-lunges", "squat45"]:

        for j in range(len(error_frame_list)):
            frame_couple = error_frame_list[j]
            user_image = "Dataset/PaddedFrames/" + exercise + "/frame" + str(frame_couple[1]) + ".jpg"
            user_sk = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl")
            user_angles, user_angles_index = retrieve_angles(user_sk, body_part="lower")
            trainer_sk = open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl")
            trainer_angles, trainer_angles_index = retrieve_angles(trainer_sk, body_part="lower")
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_angles)):
                joint_distances.append((np.linalg.norm(user_angles[i] - trainer_angles[i]), i))
            thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
            top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
            error_points = []
            for tuple in top_tier_distances:
                if tuple[0] > thr:
                    coords_idx = np.where(lower_body_names(user_sk)[tuple[1]] == user_sk["joint_names"])
                    error_points.append(tuple[1])
                    joint_error_counter[repetition_error_list[j]][coords_idx] += 1
            error_points = [user_angles_index[i] for i in error_points]
            errors = []
            errors_2d = []
            for idx in range(len(error_points)):
                errors.append(np.reshape(find_coord(user_sk, error_points[idx]), newshape=(3,)))
                errors_2d.append(np.reshape(find_2d_coord(user_sk, error_points[idx]), newshape=(3,)))
            errors = np.array(errors)
            if visualize_errors_flag:
                visualize_errors(trainer_sk, user_sk, trainer_image, user_image, errors, errors_2d, frame_couple)
    else:
        joints_number=24
        for j in range(len(error_frame_list)):
            frame_couple = error_frame_list[j]
            user_image = "Dataset/PaddedFrames/" + exercise + "/frame" + str(frame_couple[1]) + ".jpg"
            user_sk = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl")
            user_angles, user_angles_index = retrieve_angles(user_sk)
            trainer_sk = open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl")
            trainer_angles, trainer_angles_index = retrieve_angles(trainer_sk)
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_angles)):
                joint_distances.append((np.linalg.norm(user_angles[i] - trainer_angles[i]), i))
            thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
            top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
            error_points = []
            for tuple in top_tier_distances:
                if tuple[0] > thr:
                    coords_idx = np.where(user_sk["joint_names"][tuple[1]] == user_sk["joint_names"])
                    error_points.append(tuple[1])
                    joint_error_counter[repetition_error_list[j]][coords_idx] += 1
            error_points = [user_angles_index[i] for i in error_points]
            errors = []
            errors_2d = []
            for idx in range(len(error_points)):
                errors.append(np.reshape(find_coord(user_sk, error_points[idx]), newshape=(3,)))
                errors_2d.append(np.reshape(find_2d_coord(user_sk, error_points[idx]), newshape=(3,)))
            errors = np.array(errors)
            if visualize_errors_flag:
                visualize_errors(trainer_sk, user_sk, trainer_image, user_image, errors, errors_2d, frame_couple)
    MCE = np.argmax(np.sum(joint_error_counter, axis=0))
    print("L'articolazione che è stata maggiormente sbagliata nel corso dell'esercizio " + exercise[:exercise.find("_")] + " è: " + str(user_sk["joint_names"][MCE]) + " (" + str(int(np.sum(joint_error_counter, axis=0)[MCE])) + ")\tSuccesso esercizio: " + str(round((1 - (np.sum(joint_error_counter)) / (frames_number * joints_number)) * 100, 2)) + "%")
    for i in range(joint_error_counter.shape[0]):
        MCE = np.argmax(joint_error_counter[i])
        print("L'articolazione che è stata maggiormente sbagliata nel corso della ripetizione " + str(i) + " è: " + str(user_sk["joint_names"][MCE]) + " (" + str(int(joint_error_counter[i][MCE])) + ")\tSuccesso ripetizione: " + str(round((1 - (np.sum(joint_error_counter[i])) / ((frames_number / len(np.unique(repetition_error_list))) * joints_number)) * 100, 2)) + "%")
