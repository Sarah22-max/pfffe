import os
import sys
import glob

def init_spark_path():
    """Configure paths to load PySpark from SPARK_HOME."""
    spark_home = os.environ.get("SPARK_HOME", r"C:\spark\spark-3.5.3-bin-hadoop3")
    if not os.path.exists(spark_home):
        print(f"Warning: SPARK_HOME path does not exist: {spark_home}")
        return
        
    python_dir = os.path.join(spark_home, "python")
    lib_dir = os.path.join(python_dir, "lib")
    
    # Add main python folder
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
        
    # Find py4j zip file dynamically
    py4j_zips = glob.glob(os.path.join(lib_dir, "py4j-*-src.zip"))
    if py4j_zips:
        py4j_zip = py4j_zips[0]
        if py4j_zip not in sys.path:
            sys.path.insert(0, py4j_zip)
            
    # Add pyspark.zip
    pyspark_zip = os.path.join(lib_dir, "pyspark.zip")
    if os.path.exists(pyspark_zip) and pyspark_zip not in sys.path:
        sys.path.insert(0, pyspark_zip)

# Run path initialization immediately upon import
init_spark_path()

def get_spark_session(app_name="PFE_SantePublique", master="local[*]"):
    """Create and return a configured SparkSession."""
    from pyspark.sql import SparkSession
    
    # Configure spark session
    builder = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.sql.shuffle.partitions", "8")
        
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
