import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import segmentation
from scipy.io import loadmat
from sklearn import cluster, metrics
import itertools

# Input data
X = loadmat('PaviaU.mat')['paviaU']
y = loadmat('PaviaU_gt.mat')['paviaU_gt']


def params(X):
    pixels_dict = {}
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                pixels_dict[(i, j, k)] = [slic[i, j]]
    return pixels_dict


if __name__ == '__main__':
    X = np.divide(X, np.max(X))
    # Arrays for HPT job
    compactness_values = [0.05, 0.1, 0.2, 0.5]
    segments_values = [200, 500, 1000, 2000, 5000]

    # Best parameters
    # compactness_values = [0.1]
    # segments_values = [5000]
    report = open('report.txt', 'w', encoding='utf8')
    for c_value, s_value in itertools.product(compactness_values, segments_values):
        slic = segmentation.slic(X, n_segments=s_value, start_label=0, compactness=c_value)
        info = f"Compactness: {round(c_value, 3)} | Segments: {s_value} | Clusters number: {len(np.unique(slic))}"
        print(info)
        report.write(info + '\n')
        dic = {}
        l = 0
        res = np.zeros((X.shape[0], X.shape[1], 3))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if slic[i, j] not in dic.keys():
                    dic[slic[i, j]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                res[i, j, 0] = dic[slic[i, j]][0] / 255
                res[i, j, 1] = dic[slic[i, j]][1] / 255
                res[i, j, 2] = dic[slic[i, j]][2] / 255
                l += 1

        superpixels = {}
        for i in range(slic.shape[0]):
            for j in range(slic.shape[1]):
                if slic[i, j] not in superpixels.keys():
                    superpixels[slic[i, j]] = (X[i, j], 1)
                superpixels[slic[i, j]] += (X[i, j], 1)

        superpixels_vector = np.zeros((len(np.unique(slic)), X.shape[2]))
        l = 0
        for i in superpixels.keys():
            superpixels[i] = np.divide(superpixels[i][0], superpixels[i][1])
            superpixels_vector[l] = superpixels[i]
            l += 1

        kmeans = cluster.KMeans(n_clusters=10)
        kmeans.fit(superpixels_vector)
        dic3 = {}
        results = np.zeros((X.shape[0], X.shape[1], 3))
        l = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if kmeans.labels_[slic[i, j]] not in dic3:
                    dic3[kmeans.labels_[slic[i, j]]] = (
                        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                results[i, j, 0] = dic3[kmeans.labels_[slic[i, j]]][0] / 255
                results[i, j, 1] = dic3[kmeans.labels_[slic[i, j]]][1] / 255
                results[i, j, 2] = dic3[kmeans.labels_[slic[i, j]]][2] / 255
                l += 1

        final_kmeans = cluster.KMeans(n_clusters=10)
        to_fit = results.reshape(results.shape[0] * results.shape[1], 3)
        final_kmeans.fit(to_fit)
        score_point = metrics.davies_bouldin_score(to_fit, y.flatten())
        score = f"Davies-Bouldin Score: {str(round(score_point, 2))}"
        print(score)
        report.write(score + '\n\n')
        plt.title(info + '\n' + score)
        plt.imshow(results)
        output_filename = f"{info.replace(':', '_').replace('|', '-').replace('.', ',')} + {score.replace(':', '_').replace('.', ',')}"
        plt.savefig(f"{output_filename.replace(' ', '')}.png")
        plt.show()
    report.close()
