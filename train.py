from pyspark.sql import SparkSession
# from pyspark.sql.types import ArrayType, StringType
# from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from nltk.stem.lancaster import LancasterStemmer

# local mode
spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .master("local[*]") \
    .getOrCreate()

# cluster mode
# spark = SparkSession \
#     .builder \
#     .appName("IsItHelpfull") \
#     .config("spark.executor.instances", 4) \
#     .config("spark.executor.cores", 1) \
#     .getOrCreate()

data_df = spark \
    .read.parquet("output/reviews_preprocessed.parquet")

# tokenize words
tokenizer = \
    RegexTokenizer(inputCol="reviewText", outputCol="wordsRaw", pattern="\\W")
data_df_tokenized = tokenizer.transform(data_df)

# remove stop words
remover = StopWordsRemover(inputCol="wordsRaw", outputCol="words")
data_df_filtered = remover.transform(data_df_tokenized)

# skip stemming, haven't figured out how to bootstrap EMR with nltk installed
# # stemming
# stemmer = LancasterStemmer()
# stemmer_udf = udf(
#     lambda tokens: [stemmer.stem(token) for token in tokens], 
#     ArrayType(StringType())
# )
# data_df_stemmed = data_df_filtered.withColumn("wordsStemmed", stemmer_udf("words"))

# # hashing term frequency 
# hashing_term_freq = \
#     HashingTF(inputCol="wordsStemmed", outputCol="featuresRaw", numFeatures=5000)
# data_df_tf = hashing_term_freq.transform(data_df_stemmed)

# hashing term frequency 
hashing_term_freq = \
    HashingTF(inputCol="words", outputCol="featuresRaw", numFeatures=5000)
data_df_tf = hashing_term_freq.transform(data_df_filtered)

# inverse document frequency
inv_doc_freq = IDF(inputCol="featuresRaw", outputCol="features", minDocFreq=5)
inv_doc_freq_fitted = inv_doc_freq.fit(data_df_tf)
data_df_tfidf = inv_doc_freq_fitted.transform(data_df_tf)

# encode classes
indexer = StringIndexer(inputCol="category", outputCol="label")
indexer_fitted = indexer.fit(data_df_tfidf)
data_prepared_df = indexer_fitted.transform(data_df_tfidf)

# train
log_reg = LogisticRegression(
    featuresCol="features", labelCol="label", predictionCol="prediction",
    maxIter=100, regParam=0.3, elasticNetParam=0
)
log_reg_fitted = log_reg.fit(data_prepared_df)
log_reg_fitted.save("output/reviews_model.model")
