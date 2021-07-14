from numpy.linalg import eig

from skeleton_functions import *


def mean_value():
    for directory in os.listdir("Dataset/Skeletons/"):
        for frame in os.listdir("Dataset/Skeletons/" + directory):
            mean_vector = np.zeros(shape=(1,3))
            with open("Dataset/Skeletons/"+directory+"/"+frame, "rb") as skeleton_file:
                skeleton = pkl.load(skeleton_file)
            for coords in skeleton["joint_coordinates"]:
                mean_vector += coords
            mean_vector /= len(skeleton["joint_coordinates"])
            skeleton["mean_value"] = mean_vector
            with open("Dataset/Skeletons/" + directory + "/" + frame, "wb") as skeleton_file:
                pkl.dump(skeleton, skeleton_file)
    return 0


def covariance_distance(C1, C2):
    somm = 0
    eigen_vector = eig(C1, C2)[0]
    for i in range(C1.shape[0]):
        somm += pow(np.log(eigen_vector[i]), 2)
    dist = sqrt(somm)
    return dist


def sequence_cov_distance(S1, S2):
    sequence_dist = []
    dist, path = fastdtw(S1, S2)
    for idx in path:
        C1 = np.cov(np.transpose(S1[idx[0]]))
        C2 = np.cov(np.transpose(S2[idx[1]]))
        sequence_dist.append(covariance_distance(C1, C2))
    return dist, sequence_dist, path


def skeleton_covariance_matrix(skeleton):
    print(skeleton["mean_value"])
    C = np.zeros(shape = (3, 3))
    mean_vector = np.transpose(skeleton["mean_value"])
    for coords in skeleton["joint_coordinates"]:
        coords = np.reshape(coords, newshape=(3,1))
        C += np.dot(coords-mean_vector, np.transpose(coords-mean_vector))
    C /= (len(skeleton["joint_coordinates"]) - 1)
    return C