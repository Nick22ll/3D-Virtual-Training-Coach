from skeleton_functions import *

# Individua le ripetizioni nei vari esercizi
def euclidean_identify_repetitions(exercise):
    distances=[]
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        ref_sk = upper_body(open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame0.pkl"))
        for frame in range(len(os.listdir("Dataset/NormalizedSkeletons/" + exercise))):
            skeleton = upper_body(open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame" + str(frame * 5) + ".pkl"))
            distances.append(np.linalg.norm(ref_sk - skeleton))
    elif exercise[:exercise.find("_")] in [ "double-lunges", "single-lunges", "squat45"]:
        ref_sk = lower_body(open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame0.pkl"))
        for frame in range(len(os.listdir("Dataset/NormalizedSkeletons/" + exercise))):
            skeleton = lower_body(open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame" + str(frame * 5) + ".pkl"))
            distances.append(np.linalg.norm(ref_sk - skeleton))
    else:
        ref_sk = open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame0.pkl")["joint_coordinates"]
        for frame in range(len(os.listdir("Dataset/NormalizedSkeletons/" + exercise))):
            skeleton = open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame" + str(frame * 5) + ".pkl")["joint_coordinates"]
            distances.append(np.linalg.norm(ref_sk - skeleton))

    candidate_PoI=[]
    PoI=[(0,0)]
    thr = np.mean(np.array(distances))
    for i in range(len(distances)):
        if distances[i] <= thr:
            candidate_PoI.append(i*5)
    i=0
    while i < len(candidate_PoI):
        temp = []
        j=0
        while i+j+1 < len(candidate_PoI) and candidate_PoI[i+j+1]-candidate_PoI[i+j]<=10:
            temp.append(candidate_PoI[i+j])
            j+=1
        temp.append(candidate_PoI[i+j])
        if len(temp) > 1:
            previous_el = int(PoI[len(PoI)-1][1])
            PoI.append((previous_el,temp[int(len(temp)/2)]))  #aggiungo alla lista PoI l'intervallo di una certa ripetizione (0-50, 50-90, ecc...)
            i+=j
        else:
            previous_el = int(PoI[len(PoI) - 1][1])
            PoI.append((previous_el,temp[0]))
        i += 1
    PoI.pop(0)
    if PoI[0][1] <= 30:   #vincolo che evita che il primo punto di interesse sia preso entro un secondo dall'inizio dell'esercizio
        PoI.pop(0)
        PoI[0] = (0, PoI[0][1])
    if len(PoI)>1:
        PoI.pop(len(PoI)-1)
    previous_el = int(PoI[len(PoI) - 1][1])
    PoI.append((previous_el, len(os.listdir("Dataset/NormalizedSkeletons/" + exercise))*5-5))
    return PoI

def retrive_euclidean_PoI_sequences(exercise):
    PoI = euclidean_identify_repetitions(exercise)
    sequences = []
    index_list = []
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                sequence.append(upper_body(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl")))
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    elif exercise[:exercise.find("_")] in [ "double-lunges", "single-lunges", "squat45"]:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                sequence.append(lower_body(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl")))
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    else:
        for tuple in PoI:
            sequence = []
            temp_index = []
            for frame in range(tuple[0], tuple[1], 5):
                sequence.append(open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl")["joint_coordinates"])
                temp_index.append(frame)
            index_list.append(temp_index)
            sequences.append(sequence)
    return sequences, index_list


def sequence_euclidean_distance(S1, S2):
    S1 = np.reshape(S1, newshape=(len(S1), S1[0].shape[0]*S1[0].shape[1]))
    S2 = np.reshape(S2, newshape=(len(S2), S2[0].shape[0] * S2[0].shape[1]))
    sequence_dist = []
    dist, path = fastdtw(S1, S2, dist=euclidean)
    for idx in path:
        sequence_dist.append(np.linalg.norm(S1[idx[0]] - S2[idx[1]]))
    return dist/len(path), path, sequence_dist


def repetitions_euclidean_distance(exercise):
    user_sequences, user_index = retrive_euclidean_PoI_sequences(exercise)
    trainer_sequences, trainer_index = retrive_euclidean_PoI_sequences(exercise.replace(exercise[exercise.find("_"):], "_0"))
    distances = []
    i=0
    if len(user_sequences) > len(trainer_sequences):
        while i<len(trainer_sequences):
            distances.append((i,i, sequence_euclidean_distance(trainer_sequences[i],user_sequences[i] )))
            i+=1
        while i < len(user_sequences):
            distances.append((0, i, sequence_euclidean_distance(trainer_sequences[0],user_sequences[i] )))
            i+=1
    else:
        while i<len(user_sequences):
            distances.append((i,i, sequence_euclidean_distance(trainer_sequences[i], user_sequences[i])))
            i+=1
        while i < len(trainer_sequences):
            distances.append((i,0, sequence_euclidean_distance(trainer_sequences[i], user_sequences[0])))
            i+=1
    return distances, trainer_index, user_index


def identify_euclidean_errors(exercise, repetition_distance, joint_thr_multiplier = 1.0, frame_thr_multiplier= 1.0, visualize_errors_flag=True):
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
            user_coordinates = upper_body(user_sk)
            trainer_sk = open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0])+".pkl")
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            trainer_coordinates = upper_body(trainer_sk)
            joint_distances = []
            for i in range(len(user_coordinates)):
                joint_distances.append((np.linalg.norm(user_coordinates[i] - trainer_coordinates[i]), i)) #memorizza distanza e indice coordinata
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
    elif exercise[:exercise.find("_")] in [ "double-lunges", "single-lunges", "squat45"]:
        for j in range(len(error_frame_list)):
            frame_couple = error_frame_list[j]
            user_image = "Dataset/PaddedFrames/" + exercise + "/frame" + str(frame_couple[1]) + ".jpg"
            user_sk = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl")
            user_coordinates = lower_body(user_sk)
            trainer_sk = open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl")
            trainer_coordinates = lower_body(trainer_sk)
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_coordinates)):
                joint_distances.append((np.linalg.norm(user_coordinates[i] - trainer_coordinates[i]),i))
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
            user_sk = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame_couple[1]) + ".pkl")
            user_coordinates = user_sk["joint_coordinates"]
            trainer_sk = open_normalized_skeleton(exercise.replace(exercise[exercise.find("_"):], "_0") + "/normalized_skeleton_frame" + str(frame_couple[0]) + ".pkl")
            trainer_coordinates = trainer_sk["joint_coordinates"]
            trainer_image = "Dataset/PaddedFrames/" + exercise.replace(exercise[exercise.find("_"):], "_0") + "/frame" + str(frame_couple[0]) + ".jpg"
            joint_distances = []
            for i in range(len(user_coordinates)):
                joint_distances.append((np.linalg.norm(user_coordinates[i] - trainer_coordinates[i]),i))
            thr = np.mean([tup[0] for tup in joint_distances]) * joint_thr_multiplier
            top_tier_distances = sorted(joint_distances, key=lambda tup: tup[0], reverse=True)
            error_points = []
            for tuple in top_tier_distances:
                if tuple[0] > thr:
                    coords_idx = np.where(user_sk["joint_names"][tuple[1]] == user_sk["joint_names"])
                    error_points.append(tuple[1])
                    joint_error_counter[repetition_error_list[j]][coords_idx] += 1
            if visualize_errors_flag:
                visualize_errors(trainer_sk, user_sk, trainer_image, user_image, user_sk["joint_coordinates"][error_points],user_sk["joint_2d_coordinates"][error_points], frame_couple)
    MCE = np.argmax(np.sum(joint_error_counter, axis=0))
    print("L'articolazione che è stata maggiormente sbagliata nel corso dell'esercizio " + exercise[:exercise.find("_")] + " è: " + str(user_sk["joint_names"][MCE]) + " (" + str(
        int(np.sum(joint_error_counter, axis=0)[MCE])) + ")\tSuccesso esercizio: " + str(round((1 - (np.sum(joint_error_counter)) / (frames_number * joints_number)) * 100, 2)) + "%")
    for i in range(joint_error_counter.shape[0]):
        MCE = np.argmax(joint_error_counter[i])
        print("L'articolazione che è stata maggiormente sbagliata nel corso della ripetizione " + str(i) + " è: " + str(user_sk["joint_names"][MCE]) + " (" + str(
            int(joint_error_counter[i][MCE])) + ")\tSuccesso ripetizione: " + str(round((1 - (np.sum(joint_error_counter[i])) / ((frames_number / len(np.unique(repetition_error_list))) * joints_number)) * 100, 2)) + "%")
