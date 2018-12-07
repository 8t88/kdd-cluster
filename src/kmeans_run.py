from pyspark.ml.clustering import KMeans
import prep_data

def kmeans_build(dataset, seed=1024,num_steps=21,batch_size=50,num_features=12):

    vector_features = prep_data.vectorize_features(dataset)
    num_clusters = vector_features.select('index_label').distinct().count() #22

    kmeans = KMeans().setK(num_clusters).setSeed(seed)
    kmeansmodel = kmeans.fit(vector_features)

    return kmeansmodel

def sse_centers(model):
    sse = model.computeCost(vector_features)
    print("Sum of Squared Errors = " + str(sse))

    centers = model.clusterCenters()
    print ("Cluster Centers: ")
    for center in centers:
        print(center)

def datalabels(model):
    kmeans_labels = kmeansmodel.transform(vector_features)
    
    print(kmeans_labels.columns)
    kmeans_labels.collect()
    return kmeans_labels

