import numpy as np
import scipy
from scipy.linalg import fractional_matrix_power, logm
from visualize_library import *
from skeleton_functions import *
import os


def Hankel(directory, r=9, body_part="full"):
    frames_number = len(os.listdir(directory))
    s = frames_number + 1 - r
    t_list = []
    newshape = 0
    for frame in os_sorted(os.listdir(directory)):
        with open(directory + "/" + frame, "rb") as skf:
            skeleton = pkl.load(skf)
        if body_part == "upper":
            t = upper_body(skeleton)
        elif body_part == "lower":
            t = lower_body(skeleton)
        else:
            t = skeleton["joint_coordinates"]
        newshape = (t.shape[0] * t.shape[1], 1)
        t = np.reshape(t, newshape=newshape)
        t_list.append(t)

    Ht = np.empty(shape=(r * newshape[0], s), dtype=float)

    for j in range(s):
        column = list(t_list[j:j + r])
        scalar_column = []
        for i in range(r):
            scalar_column = scalar_column + list(column[i])
        scalar_column = np.array(scalar_column)
        for k in range(len(scalar_column)):
            Ht[k][j] = scalar_column[k]
    return Ht


def rank(matrix):
    return np.linalg.matrix_rank(matrix)


def isPD(x):
    return np.all(np.linalg.eigvals(x) > 0)


def isPSD(A, tol=1e-12):
    E, V = scipy.linalg.eigh(A)
    return np.all(E > -tol)


def gram_matrix(directory, sigma=10e-2, r=9, body_part="full"):
    hankel = Hankel(directory, r, body_part)
    gram = np.dot(hankel, np.transpose(hankel))
    gram = gram / np.linalg.norm(gram, "fro")
    gram = gram + (sigma * np.identity(gram.shape[0]))
    return gram


def skeleton_gram_matrix(skeleton, body_part="full"):
    if body_part == "upper":
        coords = upper_body(skeleton)
    elif body_part == "lower":
        coords = lower_body(skeleton)
    else:
        coords = skeleton["joint_coordinates"]
    return np.dot(coords, np.transpose(coords))


# Funzione che calcola la Euclidean Riemannian Metric tra due matrici
def LERM_distance(M, N):
    log = logm(M)
    diff = logm(M) - logm(N)
    return np.linalg.norm(diff, "fro")


def AIRM_distance(M, N):
    root_M = fractional_matrix_power(M, -0.5)
    prod = np.dot(root_M, N)
    prod = np.dot(prod, root_M)
    dist = np.linalg.norm(logm(prod), "fro")
    if dist < 1e-13:
        dist = 0.0
    return dist


# Usabile SOLO su matrici PD
def AIRM_distance_eigen(M, N):
    eigenvalues = np.linalg.eigvals(np.dot(np.linalg.matrix_power(M, -1), N))
    for i in range(len(eigenvalues)):
        eigenvalues[i] = pow(log(eigenvalues[i]), 2)
    return sqrt(sum(eigenvalues))


def GRAM_geodesic_distance(M, N):
    if False not in (M == N):
        return 0.0
    tr_M = np.trace(M)
    tr_N = np.trace(N)
    root_M = fractional_matrix_power(M, 0.5)
    temp = np.dot(root_M, N)
    temp = fractional_matrix_power(np.dot(temp, root_M), 0.5)
    temp = np.trace(temp)
    temp = tr_M + tr_N - (2 * temp)
    dist = pow(temp, 0.5)
    if dist < 1e-6:
        dist = 0.0
    return dist


def JBLD_distance(M, N):
    det = np.linalg.det((M + N) * 0.5)
    temp = log(det)
    temp = temp - (0.5 * log(np.linalg.det(np.dot(M, N))))
    return sqrt(temp)


# ritorna le misure di distanza tra ripetizioni di un esercizio (trainer_ID_ripetizione, user_ID_ripetizione, tripla_di_distanza)
def repetitions_GRAM_distance(exercise):
    user_sequences, user_index = retrieve_GRAM_PoI_sequences(exercise)
    trainer_sequences, trainer_index = retrieve_GRAM_PoI_sequences(exercise.replace(exercise[exercise.find("_"):], "_0"))
    distances = []
    i = 0
    if len(user_sequences) > len(trainer_sequences):
        while i < len(trainer_sequences):
            distances.append((i, i, sequence_GRAM_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(user_sequences):
            distances.append((0, i, sequence_GRAM_distance(trainer_sequences[0], user_sequences[i])))
            i += 1
    else:
        while i < len(user_sequences):
            distances.append((i, i, sequence_GRAM_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(trainer_sequences):
            distances.append((i, 0, sequence_GRAM_distance(trainer_sequences[i], user_sequences[0])))
            i += 1
    return distances, trainer_index, user_index


def retrieve_gram_sequence(sk_sequence, body_part="full"):
    sequence = []
    for skeleton in sk_sequence:
        sequence.append(skeleton_gram_matrix(skeleton, body_part=body_part))
    return sequence


def sequence_GRAM_distance(S1, S2):
    sequence_dist = []
    dist, path = fastdtw(S1, S2, dist=GRAM_geodesic_distance)
    for idx in path:
        sequence_dist.append(GRAM_geodesic_distance(S1[idx[0]], S2[idx[1]]))
    return dist / len(path), path, sequence_dist


def GRAM_identify_repetitions(exercise):
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        body_part = "upper"
    elif exercise[:exercise.find("_")] in ["double-lunges", "single-lunges", "squat45"]:
        body_part = "lower"
    else:
        body_part = "full"

    ref_sk = open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame0.pkl")
    ref_GRAM = skeleton_gram_matrix(ref_sk, body_part)
    distances = []
    for frame in range(len(os.listdir("Dataset/NormalizedSkeletons/" + exercise))):
        skeleton = open_skeleton("Dataset/NormalizedSkeletons/" + exercise + "/normalized_skeleton_frame" + str(frame * 5) + ".pkl")
        GRAM = skeleton_gram_matrix(skeleton, body_part)
        distances.append(GRAM_geodesic_distance(ref_GRAM, GRAM))

    candidate_PoI = []
    PoI = [(0, 0)]
    thr = np.mean(np.array(distances))
    for i in range(len(distances)):
        if distances[i] <= thr:
            candidate_PoI.append(i * 5)
    i = 0
    while i < len(candidate_PoI):
        temp = []
        j = 0
        while i + j + 1 < len(candidate_PoI) and candidate_PoI[i + j + 1] - candidate_PoI[i + j] <= 10:
            temp.append(candidate_PoI[i + j])
            j += 1
        temp.append(candidate_PoI[i + j])
        if len(temp) > 1:
            previous_el = int(PoI[len(PoI) - 1][1])
            PoI.append((previous_el, temp[int(len(temp) / 2)]))  # aggiungo alla lista PoI l'intervallo di una certa ripetizione (0-50, 50-90, ecc...)
            i += j
        else:
            previous_el = int(PoI[len(PoI) - 1][1])
            PoI.append((previous_el, temp[0]))
        i += 1
    PoI.pop(0)
    if PoI[0][1] <= 30:  # vincolo che evita che il primo punto di interesse sia preso entro un secondo dall'inizio dell'esercizio
        PoI.pop(0)
        PoI[0] = (0, PoI[0][1])
    if len(PoI) > 1:
        PoI.pop(len(PoI) - 1)
    previous_el = int(PoI[len(PoI) - 1][1])
    PoI.append((previous_el, len(os.listdir("Dataset/NormalizedSkeletons/" + exercise)) * 5 - 5))
    return PoI


def retrieve_GRAM_PoI_sequences(exercise):
    if exercise[:exercise.find("_")] in ["arm-clap", "dumbbell-curl"]:
        body_part = "upper"
    elif exercise[:exercise.find("_")] in ["double-lunges", "single-lunges", "squat45"]:
        body_part = "lower"
    else:
        body_part = "full"
    PoI = GRAM_identify_repetitions(exercise)
    sequences = []
    index_list = []
    for tuple in PoI:
        sequence = []
        temp_index = []
        for frame in range(tuple[0], tuple[1], 5):
            skn = open_normalized_skeleton(exercise + "/normalized_skeleton_frame" + str(frame) + ".pkl")
            GRAM = skeleton_gram_matrix(skn, body_part)
            sequence.append(GRAM)
            temp_index.append(frame)
        index_list.append(temp_index)
        sequences.append(sequence)
    return sequences, index_list


def GRAM_repetitions_distance(exercise):
    user_sequences, user_index = retrieve_GRAM_PoI_sequences(exercise)
    trainer_sequences, trainer_index = retrieve_GRAM_PoI_sequences(exercise.replace(exercise[exercise.find("_"):], "_0"))
    distances = []
    i = 0
    if len(user_sequences) > len(trainer_sequences):
        while i < len(trainer_sequences):
            distances.append((i, i, sequence_GRAM_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(user_sequences):
            distances.append((0, i, sequence_GRAM_distance(trainer_sequences[0], user_sequences[i])))
            i += 1
    else:
        while i < len(user_sequences):
            distances.append((i, i, sequence_GRAM_distance(trainer_sequences[i], user_sequences[i])))
            i += 1
        while i < len(trainer_sequences):
            distances.append((i, 0, sequence_GRAM_distance(trainer_sequences[i], user_sequences[0])))
            i += 1
    return distances, trainer_index, user_index

import PySimpleGUI as sg

def browse_window():
    to_analyze = None
    sg.theme('BlueMono')   # Add a touch of color
    # All the stuff inside your window.
    layout = [[
        sg.Text("Exercises Folder"),
        sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse()
    ],
    [
        sg.Listbox(values=[], size=(50,10), enable_events=True, key="-FILE LIST-")
    ],
    [
        sg.InputText(size=(25, 1), key='-SELECTED_FOLDER-'),
        sg.Button('Analyze', key="-ANALYZE-", button_color='green')
    ]]

    # Create the Window
    window = sg.Window('3D Virtual Training Coach', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    to_analyze=None
    while True:
        event, values = window.read()
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isdir(os.path.join(folder, f))
            ]
            window["-FILE LIST-"].update(fnames)

        elif event == "-FILE LIST-":
            try:
                exercise = values["-FILE LIST-"][0]
                window["-SELECTED_FOLDER-"].update(exercise)
            except:
                pass
        elif event == "-ANALYZE-":
            try:
                exercise = values["-SELECTED_FOLDER-"]
            except:
                pass
            download_models()
            exist_flag = False
            if os.path.exists("Dataset/Skeletons/" + exercise):
                exist_flag = True
            generate_skeletons(exercise)
            if not exist_flag:
                normalize_directory(exercise)
                skeleton_fixer()
                renamer()
            to_analyze = exercise
            break
        elif event == sg.WIN_CLOSED or event == 'Cancel':
            break
    window.close()
    return to_analyze


