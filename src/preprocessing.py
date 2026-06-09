import os
from src.spark_session import get_spark_session
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType

def clean_dataframe(df):
    """
    Cleans a Spark DataFrame:
    - Renames index column _c0 to Id.
    - Renames CI.lower and CI.upper to remove dots.
    - Standardizes types for numerical features.
    - Recalculates the prevalence rate per 100,000 population.
    """
    # 1. Rename columns
    if "_c0" in df.columns:
        df = df.withColumnRenamed("_c0", "Id")
    else:
        df = df.withColumn("Id", F.monotonically_increasing_id())
        
    df = df.withColumnRenamed("CI.lower", "CI_lower") \
           .withColumnRenamed("CI.upper", "CI_upper")
           
    # 2. Cast types
    df = df.withColumn("Year", F.col("Year").cast(IntegerType())) \
           .withColumn("Count", F.col("Count").cast(IntegerType())) \
           .withColumn("Population", F.col("Population").cast(IntegerType())) \
           .withColumn("Rate", F.col("Rate").cast(DoubleType())) \
           .withColumn("CI_lower", F.col("CI_lower").cast(DoubleType())) \
           .withColumn("CI_upper", F.col("CI_upper").cast(DoubleType()))
           
    # 3. Handle missing values (fill with 0 for numeric columns)
    df = df.na.fill({"Count": 0, "Rate": 0.0, "CI_lower": 0.0, "CI_upper": 0.0})
    
    # 4. Recalculate prevalence rate per 100,000 population
    # Handle division by zero if population is 0
    df = df.withColumn(
        "Recalculated_Rate",
        F.when(F.col("Population") > 0, F.round((F.col("Count") / F.col("Population")) * 100000, 3))
         .otherwise(0.0)
    )
    
    return df

def split_county_state(df):
    """
    Splits the dataset into county-level and state-level (California) DataFrames
    to avoid double-counting in spatial analysis.
    """
    df_state = df.filter(F.col("County") == "California")
    df_county = df.filter(F.col("County") != "California")
    return df_county, df_state

def main():
    print("=" * 60)
    print("  Etape 2 : Preparation et Nettoyage des Donnees (PySpark)")
    print("=" * 60)
    
    # Initialize Spark
    spark = get_spark_session("PFE_Preprocessing")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    
    # Output directories
    output_dir = os.path.join(base_dir, "data_clean")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[INFO] Chargement des fichiers bruts...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[ERREUR] Fichiers train.csv ou test.csv manquants dans : {base_dir}")
        spark.stop()
        return
        
    df_train_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(train_path)
    df_test_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(test_path)
    
    print(f"   Train : {df_train_raw.count():,} lignes")
    print(f"   Test  : {df_test_raw.count():,} lignes")
    
    # Clean Data
    print("\n[INFO] Nettoyage des donnees...")
    df_train_clean = clean_dataframe(df_train_raw)
    df_test_clean = clean_dataframe(df_test_raw)
    
    # Split County and State
    print("[INFO] Separation des agregats regionaux (California vs Comtes)...")
    df_train_county, df_train_state = split_county_state(df_train_clean)
    df_test_county, df_test_state = split_county_state(df_test_clean)
    
    print(f"   - Comtes Train : {df_train_county.count():,} lignes")
    print(f"   - Comtes Test  : {df_test_county.count():,} lignes")
    print(f"   - Etat Train   : {df_train_state.count():,} lignes")
    print(f"   - Etat Test    : {df_test_state.count():,} lignes")
    
    # Save Cleaned Data to Parquet (optimized for Spark ML)
    print("\n[INFO] Sauvegarde au format Parquet (pour le Machine Learning)...")
    parquet_train_dir = os.path.join(output_dir, "parquet", "train")
    parquet_test_dir = os.path.join(output_dir, "parquet", "test")
    
    df_train_county.write.mode("overwrite").parquet(parquet_train_dir)
    df_test_county.write.mode("overwrite").parquet(parquet_test_dir)
    print(f"   [SUCCESS] Parquet Train enregistre dans : {parquet_train_dir}")
    print(f"   [SUCCESS] Parquet Test enregistre dans : {parquet_test_dir}")
    
    # Save Cleaned Data to CSV (for Streamlit dashboard)
    # Coalesce to 1 partition for a single CSV file output
    print("\n[INFO] Sauvegarde au format CSV (pour le tableau de bord Streamlit)...")
    csv_train_dir = os.path.join(output_dir, "csv_train")
    csv_test_dir = os.path.join(output_dir, "csv_test")
    
    # Write to a folder and then we will rename/move it for convenience in Streamlit
    df_train_county.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_train_dir)
    df_test_county.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_test_dir)
    
    # Find the generated CSV files and copy them to clean single file names
    import shutil
    def extract_single_csv(spark_csv_dir, dest_file_path):
        for file in os.listdir(spark_csv_dir):
            if file.endswith(".csv") and file.startswith("part-"):
                shutil.copy(os.path.join(spark_csv_dir, file), dest_file_path)
                break
                
    dest_train_csv = os.path.join(output_dir, "train_clean.csv")
    dest_test_csv = os.path.join(output_dir, "test_clean.csv")
    extract_single_csv(csv_train_dir, dest_train_csv)
    extract_single_csv(csv_test_dir, dest_test_csv)
    
    # Save state-level data as well
    df_train_state.coalesce(1).write.mode("overwrite").option("header", "true").csv(os.path.join(output_dir, "csv_state_train"))
    dest_state_csv = os.path.join(output_dir, "state_clean.csv")
    extract_single_csv(os.path.join(output_dir, "csv_state_train"), dest_state_csv)
    
    # Clean up spark partition directories
    shutil.rmtree(csv_train_dir, ignore_errors=True)
    shutil.rmtree(csv_test_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(output_dir, "csv_state_train"), ignore_errors=True)
    
    print(f"   [SUCCESS] CSV Train propre : {dest_train_csv}")
    print(f"   [SUCCESS] CSV Test propre : {dest_test_csv}")
    print(f"   [SUCCESS] CSV Etat propre : {dest_state_csv}")
    
    print("\n[SUCCESS] Etape 2 terminee avec succes !")
    print("=" * 60)
    spark.stop()

if __name__ == "__main__":
    main()
