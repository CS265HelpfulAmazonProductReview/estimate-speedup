from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, rand
from pyspark.sql.types import StringType

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

def category_review(votes):
    score = votes[0]/votes[1]
    return "good" if score >= 0.8 else "else"

category_review_udf = udf(category_review, StringType())

review_df = spark \
    .read.json("data/reviews.json") \
    .where(col("helpful")[1] >= 5) \
    .withColumn("category", category_review_udf("helpful")) \
    .select("reviewText", "category") \
    .cache()

# up-sample the else category, so that #else = #good
review_df_good = review_df.where(col("category") == "good").cache()
review_df_else = review_df.where(col("category") == "else").cache()
n_good = review_df_good.count()
n_else = review_df_else.count()
fraction = float(n_good)/n_else
review_df_else_upsampled = \
    review_df_else.sample(withReplacement=True, fraction=fraction)
review_df_preprocessed = review_df_good.unionAll(review_df_else_upsampled)

review_df_preprocessed.write.parquet(
    "output/reviews_preprocessed.parquet"
)

review_preprocessed_df = spark \
    .read.parquet("output/reviews_preprocessed.parquet")
review_preprocessed_df.show()
review_preprocessed_df.groupBy("category").count().show()
