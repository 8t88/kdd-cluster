import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def run_pca(vector_features, k=3):
    from pyspark.ml.feature import PCA

    feature_vec = vector_features.select('features')
    pca = PCA(k, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(feature_vec)

    result = model.transform(feature_vec).select("pcaFeatures")
    return result


def plot_3d():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    y_num_label = data['index_label']

    #resultfeats[0], resultfeats[1], resultfeats[2]
    #working on it, just temporary for now
    xnums = [reslist[x][0] for x in range(0,len(reslist))]
    ynums = [reslist[y][1] for y in range(0,len(reslist))]
    znums = [reslist[z][2] for z in range(0,len(reslist))]

    #resultfeats = result.show(truncate=False)
    mplt = plt.figure(figsize=(12,10)).gca(projection='3d')
    mplt.scatter(xnums, ynums, znums, c=y_num_label)
    mplt.set_xlabel('x')
    mplt.set_ylabel('y')
    mplt.set_zlabel('z')
    plt.legend()
    plt.show()


def barplot():




