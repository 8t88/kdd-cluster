from pyspark.ml.clustering import KMeans
from prep_data import get_kdd_data




from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

df = get_kdd_data('KDDTest+.csv')
df_non_null = df.fillna(0)

feature_labels = ['label', 'index_label']
feature_names = [x for x in df_one_hot.columns if x not in feature_labels]
print(len(feature_names))

vecAssembler = VectorAssembler(inputCols=feature_names, outputCol="features")
vector_features = vecAssembler.transform(df_non_null).drop(*feature_names)

seed = 1024
num_clusters = vector_features.select('index_label').distinct().count() #22
num_steps = 21
batch_size = 1024
num_features = 41


kmeans = KMeans().setK(num_clusters).setSeed(seed)
kmeansmodel = kmeans.fit(vector_features)

sse = kmeansmodel.computeCost(vector_features)
print("Sum of Squared Errors = " + str(sse))

centers = kmeansmodel.clusterCenters()
print ("Cluster Centers: ")
for center in centers:
    print(center)

