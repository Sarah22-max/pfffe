import os
from src.spark_session import get_spark_session
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def get_top_diseases(df, limit=15):
    """Returns top diseases by total count of cases."""
    return df.groupBy("Disease") \
             .agg(F.sum("Count").alias("Total_Cases"),
                  F.round(F.mean("Recalculated_Rate"), 2).alias("Avg_Prevalence_Rate")) \
             .orderBy(F.desc("Total_Cases")) \
             .limit(limit)

def get_yearly_trends(df, disease=None):
    """Returns total cases and average rate per year, optionally filtered by disease."""
    if disease:
        df = df.filter(df.Disease == disease)
    return df.groupBy("Year") \
             .agg(F.sum("Count").alias("Total_Cases"),
                  F.round(F.mean("Recalculated_Rate"), 2).alias("Avg_Rate")) \
             .orderBy("Year")

def get_demographic_distribution(df, disease=None):
    """Returns gender distribution of cases, optionally filtered by disease."""
    if disease:
        df = df.filter(df.Disease == disease)
        
    # Exclude Total or other potential non-gender aggregates if present
    df = df.filter(F.col("Sex").isin(["Male", "Female"]))
    
    # Calculate total cases in this subset for percentage
    total_row = df.select(F.sum("Count")).collect()
    total_cases = total_row[0][0] if total_row and total_row[0][0] is not None else 1
    
    return df.groupBy("Sex") \
             .agg(F.sum("Count").alias("Total_Cases"),
                  F.round(F.mean("Recalculated_Rate"), 2).alias("Avg_Rate"),
                  F.round((F.sum("Count") / F.lit(total_cases)) * 100, 2).alias("Percentage")) \
             .orderBy(F.desc("Total_Cases"))

def get_county_hotspots(df, disease, limit=10):
    """Returns the top counties with the highest average prevalence rate for a given disease."""
    return df.filter(df.Disease == disease) \
             .groupBy("County") \
             .agg(F.sum("Count").alias("Total_Cases"),
                  F.round(F.mean("Recalculated_Rate"), 2).alias("Avg_Rate")) \
             .orderBy(F.desc("Avg_Rate")) \
             .limit(limit)

def get_fastest_growing_diseases(df):
    """
    Computes growth rate of diseases.
    Compares the average case counts in the early years (2001-2005)
    to the late years (2010-2014) to calculate growth.
    """
    # Filter for early and late periods
    early_period = df.filter((df.Year >= 2001) & (df.Year <= 2005))
    late_period = df.filter((df.Year >= 2010) & (df.Year <= 2014))
    
    early_agg = early_period.groupBy("Disease").agg(F.sum("Count").alias("Cases_2001_2005"))
    late_agg = late_period.groupBy("Disease").agg(F.sum("Count").alias("Cases_2010_2014"))
    
    # Join and compute percentage change
    growth_df = early_agg.join(late_agg, "Disease", "inner") \
                         .withColumn("Absolute_Increase", F.col("Cases_2010_2014") - F.col("Cases_2001_2005")) \
                         .withColumn("Percentage_Growth", 
                                     F.round(((F.col("Cases_2010_2014") - F.col("Cases_2001_2005")) / 
                                              F.when(F.col("Cases_2001_2005") > 0, F.col("Cases_2001_2005")).otherwise(1)) * 100, 2)) \
                         .orderBy(F.desc("Percentage_Growth"))
    return growth_df

def main():
    print("=" * 60)
    print("  Etape 3 : Analyses Statistiques de Sante Publique (PySpark)")
    print("=" * 60)
    
    # Initialize Spark Session
    spark = get_spark_session("PFE_Analytics")
    
    # Load cleaned data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet_path = os.path.join(base_dir, "data_clean", "parquet", "train")
    
    if not os.path.exists(parquet_path):
        print(f"[ERREUR] Le dossier de donnees nettoyees n'existe pas : {parquet_path}")
        print("Veuillez d'abord executer preprocessing.py.")
        spark.stop()
        return
        
    df = spark.read.parquet(parquet_path)
    
    # 1. Top Diseases
    print("\n>>> TOP 10 DES MALADIES PAR NOMBRE TOTAL DE CAS :")
    top_diseases = get_top_diseases(df, 10)
    top_diseases.show(10, False)
    
    # 2. Demographic Disparities (HIV as example)
    print("\n>>> DISPARITES TEMPORELLES ET DEMOGRAPHIQUES (HIV) :")
    hiv_dem = get_demographic_distribution(df, "HIV")
    hiv_dem.show()
    
    # 3. County Hotspots (Chlamydia as example)
    print("\n>>> TOP 5 DES COMTES HOTSPOTS (Chlamydia) :")
    chlamydia_hotspots = get_county_hotspots(df, "Chlamydia", 5)
    chlamydia_hotspots.show()
    
    # 4. Growth Trends
    print("\n>>> TOP 10 DES MALADIES AVEC LA PLUS FORTE CROISSANCE (2001-2005 vs 2010-2014) :")
    growing_diseases = get_fastest_growing_diseases(df)
    growing_diseases.show(10, False)
    
    print("\n[SUCCESS] Etape 3 terminee avec succes !")
    print("=" * 60)
    spark.stop()

if __name__ == "__main__":
    main()
