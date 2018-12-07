import pandas as pd
import numpy as np
from pyspark.ml.feature import OneHotEncoder, StringIndexer
def get_kdd_data(csv):
    """
    input the filepath of the csv data, expecting this to be in the kdd format,
    with the columns in the standard order
    """
    col_names = np.loadtxt('./col_names.txt', dtype='str')
    data = pd.read_csv(csv, header=None, names=col_names)

    #want to change this so spark reads the csv straight, no pd mediation
    df = spark.createDataFrame(data)

    #one-hot encode the string columns
    one_hot_cols = ['protocol_type', 'service', 'flag']
    for label in one_hot_cols:
        stringIndexer = StringIndexer(inputCol=label, outputCol=label+"_index")
        model = stringIndexer.fit(df)
        indexed = model.transform(df)

        encoder = OneHotEncoder(inputCol=label+"_index", outputCol=label+"_vec")
        encoded = encoder.transform(indexed)
        df = encoder.transform(indexed).drop(label+"_index").drop(label)
    
    return df

def vectorize_features(df, feature_labels = ['label', 'index_label']):
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import VectorAssembler

    df_non_null = df.fillna(0)

    feature_names = [x for x in df_one_hot.columns if x not in feature_labels]
    
    vecAssembler = VectorAssembler(inputCols=feature_names, outputCol="features")
    vector_features = vecAssembler.transform(df_non_null).drop(*feature_names)
    return vector_features

def runPCA(vector_features, k=3):
    from pyspark.ml.feature import PCA

    #convert df to feature_vec
    feature_vec = vector_features.select('features')
    pca = PCA(k, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(feature_vec)

    result = model.transform(feature_vec).select("pcaFeatures")
    return result
