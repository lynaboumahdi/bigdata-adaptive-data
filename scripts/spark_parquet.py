from pyspark.sql import SparkSession


spark = SparkSession.builder \
.master("local[*]") \
.appName("FashionMNIST") \
.getOrCreate()


df=spark.read.csv(
"data/canonical/fashion_mnist.csv",
header=True,
inferSchema=True
)


print(df.count())


df.write.mode("overwrite").parquet(
"data/parquet"
)


spark.stop()