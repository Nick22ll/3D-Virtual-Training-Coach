import os
import time

from GramDistances import *
from visualize_library import *
from EuclideanDistances import *
from CombinedDistances import *
from AnglesDistances import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

def main():
    try_me()
    return

def try_me():
    analyze_window(browse_window())
    return



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
        sg.Button('Next', key="-NEXT-", button_color='green')
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
        elif event == "-NEXT-":
            exercise = values["-SELECTED_FOLDER-"]
            if exercise != '':
                download_models()
                exist_flag = False
                if os.path.exists("Dataset/Skeletons/" + exercise):
                    exist_flag = True
                generate_skeletons(exercise)
                if not exist_flag:
                    time.sleep(0.1)
                    normalize_directory(exercise)
                    skeleton_fixer()
                    renamer()
                to_analyze = exercise
                break
        elif event == sg.WIN_CLOSED or event == 'Cancel':
            break
    window.close()
    return to_analyze


def analyze_window(exercise):
    if exercise == None:
        return
    sg.theme('BlueMono')  # Add a touch of color
    # All the stuff inside your window.
    layout = [[
        sg.Text("Evaluation Metric for Skeletons Pose Errors"),
        sg.Combo(["Angles Metric", "Euclidean Metric", "Combined Metric", "GRAM Metric"], enable_events=True, default_value='Angles Metric', key="REP_DISTANCE")
    ],
    [
        sg.Text("Evaluation Metric for Joint Errors"),
        sg.Combo(["Angles Metric", "Euclidean Metric", "Combined Metric"], enable_events=True,default_value='Angles Metric', key="POSE_METRIC")
    ],
    [
        sg.Text("Pose Error Threshold Multiplier"),
        sg.Slider(range=(1.0, 2.5), size=(55,20), orientation="horizontal", resolution=0.1, tick_interval=0.1, default_value=1.5, enable_events=True, key="POSE_THR")
    ],
    [
        sg.Text("Joint Error Threshold Multiplier"),
        sg.Slider(range=(1.0, 2.5), size=(55,20), orientation="horizontal", resolution=0.1, tick_interval=0.1, default_value=1.5, enable_events=True, key="JOINT_THR")
    ],
    [
        [sg.Checkbox('Visualize Errors', default=True, key="VISUALIZE_ERRORS")]
    ],
    [
        sg.Button('Analyze', key="ANALYZE", button_color='green')
    ]]

    # Create the Window
    window = sg.Window('3D Virtual Training Coach', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    to_analyze = None
    while True:
        event, values = window.read()
        if event == "ANALYZE":
            rep_distance = None
            if values["REP_DISTANCE"] == "Angles Metric":
                rep_distance = repetitions_angles_distance
            if values["REP_DISTANCE"] == "Euclidean Metric":
                rep_distance = repetitions_euclidean_distance
            if values["REP_DISTANCE"] == "Combined Metric":
                rep_distance = repetitions_combined_distance
            if values["REP_DISTANCE"] == "GRAM Metric":
                rep_distance = repetitions_GRAM_distance

            if values["POSE_METRIC"] == "Angles Metric":
                identify_angles_errors(exercise, rep_distance, values["JOINT_THR"], values["POSE_THR"], values["VISUALIZE_ERRORS"])
            if values["POSE_METRIC"] == "Euclidean Metric":
                identify_euclidean_errors(exercise, rep_distance, values["JOINT_THR"], values["POSE_THR"], values["VISUALIZE_ERRORS"])
            if values["POSE_METRIC"] == "Combined Metric":
                identify_combined_errors(exercise, rep_distance, values["JOINT_THR"], values["POSE_THR"], values["VISUALIZE_ERRORS"])
        elif event == sg.WIN_CLOSED or event == 'Cancel':
            break
    window.close()


if __name__ == '__main__':
    main()

