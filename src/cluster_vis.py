import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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


def barplot(kmeans_labels):
    label_prediction_agg_pd = kmeans_labels.groupBy("index_label", "prediction").count().sort("count").toPandas()

    label_agg_pd = kmeans_labels.groupBy("index_label").count().sort("count").toPandas()
    prediction_agg_pd = kmeans_labels.groupBy("prediction").count().sort("count").toPandas()

    y_pos = np.arange(len(label_agg_pd['index_label']))

    plt.subplot(2,1,1)
    plt.bar(y_pos, label_agg_pd['count'], align='center', alpha=0.5)
    plt.xticks(y_pos, label_agg_pd['index_label'])
    plt.title('Count of Labels')

    plt.subplot(2,1,2)
    plt.bar(y_pos, prediction_agg_pd['count'], align='center', alpha=0.5)
    plt.xticks(y_pos, prediction_agg_pd['prediction'])
    plt.title('Count of Cluster Predictions')
    plt.show()


def scatterplots():




