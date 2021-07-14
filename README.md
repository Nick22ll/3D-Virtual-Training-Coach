# 3D-Virtual-Training-Coach
This repository contains code for the methods described in the following [paper](https://github.com/Nick22ll/Virtual-Training-Coach/blob/main/Relazione.pdf).<br>
The code extract 3D skeletons through [MeTRAbs](https://github.com/isarandi/metrabs) from homemade videos frames to establish which parts of a fitness exercise are not correctly executed.

## System Requirements
For GPU enviroment:
* CUDA 11.0 or greater versions;
* TensorFlow-gpu 2.5.0 or greater versions.

For CPU enviroment:
* TensorFlow 2.5.0

To generate 3D skeletons is **necessary** to download some pre-trained models: before running the code, you have to launch the method **download_models()**.

## Running the Code
First of all you have to generate 3D skeletons of fitness exercise frames running the **generate_dataset( *frames_path* )** method: make sure that the **Frames** directory contains a sub-directory of the specific exercise frames, i.e *Frames/arm-clap_1/frame.jpg*.<br>
Then normalize your 3D skeletons through **normalize_dataset()** method.<br>
Finally choose an exercise to analyze with one of the metrics defined in the [paper](https://github.com/Nick22ll/Virtual-Training-Coach/blob/main/RelazioneIVA.pdf) launching one of the following methods:
* **identify_euclidean_errors()**;
* **identify_angles_errors()**;
* **identify_combined_errors()**.

## Running Sample
You can try it in action using the **try_me()** method: it will guide you to run the code in a user friendly way with a minimal graphic interface.
<p align="center"><img src=Images/BrowseWindow.png width="60%"></p>
How to use:
1. Select the directory containing the frame exercises directories;
2. Choose an exercise from the list;
3. Click on Next!
<p align="center"><img src=Images/AnalyzeWindow.png width="60%"></p>
4. Select the metrics;
5. Set the thresholds;
6. Decide if you want to visualize the images of the errors;
7. Click on Analyze!
