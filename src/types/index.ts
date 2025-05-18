
export interface AirQualityData {
  Date: string;
  PM25: number;
  PM10: number;
  NO2: number;
  SO2: number;
  CO: number;
  O3: number;
  Temperature: number;
  Humidity: number;
  WindSpeed: number;
  AQI: number;
}

export interface ProcessedData extends AirQualityData {
  Month?: number;
  DayOfWeek?: number;
  PM_Ratio?: number;
  PM25_lag1?: number;
  NO2_lag1?: number;
}

export interface ModelResults {
  modelName: string;
  mae: number;
  rmse: number;
  r2: number;
  predictions: number[];
  actual: number[];
  featureImportance?: { feature: string; importance: number }[];
}

export interface DatasetStats {
  rowCount: number;
  columnCount: number;
  missingValueCount: number;
  duplicateCount: number;
  outlierCount: number;
}
