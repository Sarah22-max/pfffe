import os
from src.spark_session import get_spark_session
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def build_pipeline():
    # String indexers for categoricals
    disease_indexer = StringIndexer(inputCol="Disease", outputCol="Disease_Index", handleInvalid="keep")
    county_indexer = StringIndexer(inputCol="County", outputCol="County_Index", handleInvalid="keep")
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_Index", handleInvalid="keep")
    
    # One hot encoders
    encoder = OneHotEncoder(
        inputCols=["Disease_Index", "County_Index", "Sex_Index"],
        outputCols=["Disease_Vec", "County_Vec", "Sex_Vec"]
    )
    
    # Vector assembler
    assembler = VectorAssembler(
        inputCols=["Year", "Disease_Vec", "County_Vec", "Sex_Vec", "Population"],
        outputCol="features"
    )
    
    # Model - MaxBins set to 100 since we have 59 counties and many diseases (exceeds default 32)
    rf = RandomForestRegressor(
        featuresCol="features", 
        labelCol="Recalculated_Rate", 
        numTrees=20, 
        maxDepth=8, 
        seed=42, 
        maxBins=100
    )
    
    # Pipeline
    pipeline = Pipeline(stages=[disease_indexer, county_indexer, sex_indexer, encoder, assembler, rf])
    return pipeline

def main():
    print("=" * 60)
    print("  Etape 4 : Apprentissage Automatique avec PySpark MLlib")
    print("=" * 60)
    
    spark = get_spark_session("PFE_Modeling")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_parquet = os.path.join(base_dir, "data_clean", "parquet", "train")
    test_parquet = os.path.join(base_dir, "data_clean", "parquet", "test")
    
    if not os.path.exists(train_parquet) or not os.path.exists(test_parquet):
        print("[ERREUR] Donnees d'entrainement/test introuvables. Executez preprocessing.py d'abord.")
        spark.stop()
        return
        
    df_train = spark.read.parquet(train_parquet)
    df_test = spark.read.parquet(test_parquet)
    
    print(f"[INFO] Nombre d'enregistrements pour l'entrainement : {df_train.count():,}")
    print(f"[INFO] Nombre d'enregistrements pour l'evaluation  : {df_test.count():,}")
    
    # Build and train model
    print("\n[INFO] Entrainement du modele Random Forest Regressor...")
    pipeline = build_pipeline()
    model = pipeline.fit(df_train)
    
    # Evaluate
    print("[INFO] Evaluation sur le jeu de test...")
    predictions = model.transform(df_test)
    
    # Evaluate RMSE
    evaluator_rmse = RegressionEvaluator(labelCol="Recalculated_Rate", predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    
    # Evaluate R2
    evaluator_r2 = RegressionEvaluator(labelCol="Recalculated_Rate", predictionCol="prediction", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)
    
    print(f"   - Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"   - Coefficient de determination (R2) : {r2:.4f}")
    
    # Add Risk Category based on predictions
    # Basse Priorite: <= 5.0, Moyenne Priorite: 5.0 to 50.0, Haute Priorite: > 50.0
    predictions_with_risk = predictions.withColumn(
        "prediction_clean",
        F.when(F.col("prediction") < 0.0, 0.0).otherwise(F.round(F.col("prediction"), 2))
    ).withColumn(
        "Priority_Level",
        F.when(F.col("prediction_clean") <= 5.0, "Basse Priorite")
         .when((F.col("prediction_clean") > 5.0) & (F.col("prediction_clean") <= 50.0), "Moyenne Priorite")
         .otherwise("Haute Priorite")
    )
    
    # Save Model
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    model_save_path = os.path.join(base_dir, "models", "spark_rf_model")
    model.write().overwrite().save(model_save_path)
    print(f"\n[SUCCESS] Modele enregistre dans : {model_save_path}")
    
    # Save test predictions as CSV for Streamlit loading
    print("[INFO] Sauvegarde des predictions de test pour Streamlit...")
    predictions_clean = predictions_with_risk.select(
        "Id", "Disease", "County", "Year", "Sex", "Count", "Population", "Recalculated_Rate", "prediction_clean", "Priority_Level"
    ).withColumnRenamed("prediction_clean", "Predicted_Rate")
    
    predictions_csv_dir = os.path.join(base_dir, "data_clean", "predictions_tmp")
    predictions_clean.coalesce(1).write.mode("overwrite").option("header", "true").csv(predictions_csv_dir)
    
    # Find and rename single CSV file
    import shutil
    dest_predictions_csv = os.path.join(base_dir, "data_clean", "predictions_test.csv")
    for file in os.listdir(predictions_csv_dir):
        if file.endswith(".csv") and file.startswith("part-"):
            shutil.copy(os.path.join(predictions_csv_dir, file), dest_predictions_csv)
            break
    shutil.rmtree(predictions_csv_dir, ignore_errors=True)
    print(f"   [SUCCESS] Fichier de predictions enregistre dans : {dest_predictions_csv}")
    
    print("\n[SUCCESS] Etape 4 terminee avec succes !")
    print("=" * 60)
    spark.stop()

if __name__ == "__main__":
    main()
