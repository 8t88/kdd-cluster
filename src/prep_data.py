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

