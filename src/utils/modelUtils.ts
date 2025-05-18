
import { ProcessedData, ModelResults } from "@/types";

// Mock implementations of Random Forest and XGBoost for frontend demo
// In a real Python app, these would use scikit-learn and XGBoost

/**
 * Train a Random Forest model and return results
 */
export const trainRandomForestModel = (
  trainData: ProcessedData[],
  testData: ProcessedData[]
): ModelResults => {
  // Mock prediction function using a simple linear combination of features
  // In real Python, this would be a trained Random Forest model
  const predictAQI = (row: ProcessedData): number => {
    // Simple weighted combination of features to simulate a model
    return (
      row.PM25 * 0.35 +
      row.PM10 * 0.25 +
      row.NO2 * 0.2 +
      row.O3 * 0.1 +
      row.SO2 * 0.05 +
      row.CO * 0.05 +
      Math.sin(row.Month! / 12 * Math.PI * 2) * 0.1 // Seasonal factor
    );
  };

  // Make predictions
  const predictions = testData.map(row => predictAQI(row));
  const actual = testData.map(row => row.AQI);
  
  // Calculate metrics
  const { mae, rmse, r2 } = calculateMetrics(predictions, actual);

  // Calculate feature importance (mock)
  const featureImportance = [
    { feature: "PM2.5", importance: 0.35 },
    { feature: "PM10", importance: 0.25 },
    { feature: "NO2", importance: 0.20 },
    { feature: "O3", importance: 0.10 },
    { feature: "SO2", importance: 0.05 },
    { feature: "CO", importance: 0.05 }
  ];
  
  return {
    modelName: "Random Forest",
    mae,
    rmse,
    r2,
    predictions,
    actual,
    featureImportance
  };
};

/**
 * Train an XGBoost model and return results
 */
export const trainXGBoostModel = (
  trainData: ProcessedData[],
  testData: ProcessedData[]
): ModelResults => {
  // Mock prediction function with slightly different weights
  // In real Python, this would be a trained XGBoost model
  const predictAQI = (row: ProcessedData): number => {
    // XGBoost typically achieves better results, so we'll simulate that
    return (
      row.PM25 * 0.4 +
      row.PM10 * 0.25 +
      row.NO2 * 0.15 +
      row.O3 * 0.1 +
      row.SO2 * 0.05 +
      row.CO * 0.05 +
      Math.sin(row.Month! / 12 * Math.PI * 2) * 0.08 + // Seasonal factor
      (row.PM_Ratio || 0) * 0.12 // XGBoost benefits from engineered features
    );
  };

  // Make predictions
  const predictions = testData.map(row => predictAQI(row));
  const actual = testData.map(row => row.AQI);
  
  // Calculate metrics
  const { mae, rmse, r2 } = calculateMetrics(predictions, actual);

  // Calculate feature importance (mock)
  const featureImportance = [
    { feature: "PM2.5", importance: 0.40 },
    { feature: "PM10", importance: 0.25 },
    { feature: "NO2", importance: 0.15 },
    { feature: "O3", importance: 0.10 },
    { feature: "PM_Ratio", importance: 0.12 },
    { feature: "Month", importance: 0.08 },
    { feature: "SO2", importance: 0.05 },
    { feature: "CO", importance: 0.05 }
  ];
  
  return {
    modelName: "XGBoost",
    mae,
    rmse,
    r2,
    predictions,
    actual,
    featureImportance
  };
};

/**
 * Calculate evaluation metrics
 */
export const calculateMetrics = (
  predictions: number[],
  actual: number[]
): { mae: number; rmse: number; r2: number } => {
  // Mean Absolute Error
  const mae = predictions.reduce((sum, pred, i) => sum + Math.abs(pred - actual[i]), 0) / predictions.length;
  
  // Root Mean Squared Error
  const mse = predictions.reduce((sum, pred, i) => sum + Math.pow(pred - actual[i], 2), 0) / predictions.length;
  const rmse = Math.sqrt(mse);
  
  // R-squared (coefficient of determination)
  const mean = actual.reduce((sum, val) => sum + val, 0) / actual.length;
  const totalSS = actual.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
  const residualSS = predictions.reduce((sum, pred, i) => sum + Math.pow(actual[i] - pred, 2), 0);
  const r2 = 1 - (residualSS / totalSS);
  
  return { mae, rmse, r2 };
};

/**
 * Calculate correlation matrix
 */
export const calculateCorrelationMatrix = (data: ProcessedData[]): Record<string, Record<string, number>> => {
  const numericColumns = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'AQI'];
  const correlations: Record<string, Record<string, number>> = {};
  
  // Initialize correlation matrix
  numericColumns.forEach(col1 => {
    correlations[col1] = {};
    numericColumns.forEach(col2 => {
      correlations[col1][col2] = 0;
    });
  });
  
  // Calculate correlation for each pair
  numericColumns.forEach(col1 => {
    numericColumns.forEach(col2 => {
      // For demo purposes, we'll use a simplified correlation calculation
      // In a real app, we'd use Pearson correlation coefficient
      
      // Extract values
      const values1 = data.map(row => row[col1 as keyof ProcessedData] as number);
      const values2 = data.map(row => row[col2 as keyof ProcessedData] as number);
      
      // Calculate mean
      const mean1 = values1.reduce((sum, val) => sum + val, 0) / values1.length;
      const mean2 = values2.reduce((sum, val) => sum + val, 0) / values2.length;
      
      // Calculate covariance and variances
      let covariance = 0;
      let variance1 = 0;
      let variance2 = 0;
      
      for (let i = 0; i < values1.length; i++) {
        const diff1 = values1[i] - mean1;
        const diff2 = values2[i] - mean2;
        
        covariance += diff1 * diff2;
        variance1 += diff1 * diff1;
        variance2 += diff2 * diff2;
      }
      
      // Calculate correlation
      const correlation = covariance / (Math.sqrt(variance1) * Math.sqrt(variance2));
      correlations[col1][col2] = isNaN(correlation) ? 0 : correlation;
    });
  });
  
  return correlations;
};
