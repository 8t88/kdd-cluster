import prep_data
import kmeans_run
import cluster_vis


#TODO: modify to take in cmd line inputs

#setting up the parameters
trainset = '../data/KDDTrain+.csv'
testset = '../data/KDDTest+.csv'
seed = 1024
#num_clusters = vector_features.select('index_label').distinct().count() #22
num_steps = 21
batch_size = 1024
num_features = 41


kmeans_model = kmeans_run.kmeans_build(trainset,seed,num_steps, batch_size,num_features)

kmeans_run.sse_centers(kmeans_model)

kmeans_run.datalabels(kmeans_model)

#PCA
#cluster_vis barcharts, 3d plots, scatterplots
