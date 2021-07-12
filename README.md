# 3D-Virtual-Training-Coach
This repository contains code for the methods described in the following [paper](https://github.com/Nick22ll/Virtual-Training-Coach/blob/main/RelazioneIVA.pdf).<br>
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
You can try it in action using the **try_me( *frames_path* )**:make sure that the **Frames** directory contains a sub-directory of the specific exercise frames, i.e *Frames/arm-clap_1/frame.jpg*.
