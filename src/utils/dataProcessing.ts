
import { AirQualityData, ProcessedData, DatasetStats } from "@/types";

/**
 * Calculate dataset stats like missing values, duplicates, etc.
 */
export const calculateDatasetStats = (data: any[]): DatasetStats => {
  // Calculate basic stats
  const rowCount = data.length;
  const columnCount = Object.keys(data[0] || {}).length;
  
  // Count missing values
  let missingValueCount = 0;
  data.forEach(row => {
    Object.values(row).forEach(value => {
      if (value === null || value === undefined || value === '') {
        missingValueCount++;
      }
    });
  });
  
  // Find duplicates (approximation - in real Python we'd use more sophisticated methods)
  const stringifiedRows = data.map(row => JSON.stringify(row));
  const uniqueRows = new Set(stringifiedRows);
  const duplicateCount = rowCount - uniqueRows.size;
  
  // Count outliers (approximation - in real Python we'd calculate IQR)
  // For this demo, we'll just count values more than 3 standard deviations from mean
  let outlierCount = 0;
  const numericColumns = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'AQI'];
  
  numericColumns.forEach(column => {
    const values = data.map(row => Number(row[column])).filter(val => !isNaN(val));
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    values.forEach(val => {
      if (Math.abs(val - mean) > 3 * stdDev) {
        outlierCount++;
      }
    });
  });
  
  return {
    rowCount,
    columnCount,
    missingValueCount,
    duplicateCount,
    outlierCount
  };
};

/**
 * Handle missing values using median imputation
 */
export const imputeMissingValues = (data: AirQualityData[]): AirQualityData[] => {
  const numericColumns = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'AQI'];
  const medians: Record<string, number> = {};
  
  // Calculate medians
  numericColumns.forEach(column => {
    const validValues = data
      .map(row => row[column as keyof AirQualityData] as number)
      .filter(val => val !== null && val !== undefined && !isNaN(val));
      
    validValues.sort((a, b) => a - b);
    const mid = Math.floor(validValues.length / 2);
    const median = validValues.length % 2 === 0
      ? (validValues[mid - 1] + validValues[mid]) / 2
      : validValues[mid];
      
    medians[column] = median;
  });
  
  // Apply median imputation
  return data.map(row => {
    const newRow = { ...row };
    numericColumns.forEach(column => {
      const key = column as keyof AirQualityData;
      if (newRow[key] === null || newRow[key] === undefined || isNaN(Number(newRow[key]))) {
        (newRow[key] as number) = medians[column];
      }
    });
    return newRow;
  });
};

/**
 * Remove duplicate rows
 */
export const removeDuplicates = (data: AirQualityData[]): AirQualityData[] => {
  const uniqueRows = new Map();
  
  data.forEach(row => {
    // Use Date + pollutant values as a key
    const key = row.Date + 
      String(row.PM25) + 
      String(row.PM10) + 
      String(row.NO2) + 
      String(row.SO2) + 
      String(row.CO) + 
      String(row.O3);
    
    // Keep the first occurrence of each unique key
    if (!uniqueRows.has(key)) {
      uniqueRows.set(key, row);
    }
  });
  
  return Array.from(uniqueRows.values());
};

/**
 * Treat outliers using IQR method
 */
export const treatOutliers = (data: AirQualityData[]): AirQualityData[] => {
  const numericColumns = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'AQI'];
  const q1s: Record<string, number> = {};
  const q3s: Record<string, number> = {};
  const iqrs: Record<string, number> = {};
  
  // Calculate Q1, Q3 and IQR for each numeric column
  numericColumns.forEach(column => {
    const values = data
      .map(row => row[column as keyof AirQualityData] as number)
      .filter(val => val !== null && val !== undefined && !isNaN(val))
      .sort((a, b) => a - b);
    
    const q1Index = Math.floor(values.length * 0.25);
    const q3Index = Math.floor(values.length * 0.75);
    
    q1s[column] = values[q1Index];
    q3s[column] = values[q3Index];
    iqrs[column] = q3s[column] - q1s[column];
  });
  
  // Apply IQR capping
  return data.map(row => {
    const newRow = { ...row };
    
    numericColumns.forEach(column => {
      const key = column as keyof AirQualityData;
      const value = newRow[key] as number;
      
      if (value !== null && value !== undefined && !isNaN(value)) {
        const lowerBound = q1s[column] - 1.5 * iqrs[column];
        const upperBound = q3s[column] + 1.5 * iqrs[column];
        
        if (value < lowerBound) {
          (newRow[key] as number) = lowerBound;
        } else if (value > upperBound) {
          (newRow[key] as number) = upperBound;
        }
      }
    });
    
    return newRow;
  });
};

/**
 * Normalize data using Min-Max scaling
 */
export const normalizeData = (data: AirQualityData[]): AirQualityData[] => {
  const numericColumns = ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'AQI'];
  const mins: Record<string, number> = {};
  const maxs: Record<string, number> = {};
  
  // Find min and max values
  numericColumns.forEach(column => {
    const values = data
      .map(row => row[column as keyof AirQualityData] as number)
      .filter(val => val !== null && val !== undefined && !isNaN(val));
    
    mins[column] = Math.min(...values);
    maxs[column] = Math.max(...values);
  });
  
  // Apply Min-Max scaling
  return data.map(row => {
    const newRow = { ...row };
    
    numericColumns.forEach(column => {
      const key = column as keyof AirQualityData;
      const value = newRow[key] as number;
      
      if (value !== null && value !== undefined && !isNaN(value)) {
        const range = maxs[column] - mins[column];
        if (range !== 0) {
          (newRow[key] as number) = ((value - mins[column]) / range);
        } else {
          (newRow[key] as number) = 0; // If min=max, set to 0
        }
      }
    });
    
    return newRow;
  });
};

/**
 * Add feature engineering columns
 */
export const engineerFeatures = (data: AirQualityData[]): ProcessedData[] => {
  // Convert date and extract month/day
  const processed: ProcessedData[] = data.map(row => {
    const date = new Date(row.Date);
    
    return {
      ...row,
      Month: date.getMonth() + 1, // 1-12
      DayOfWeek: date.getDay(), // 0-6 (0 is Sunday)
      PM_Ratio: row.PM25 / (row.PM10 || 1) // Avoid division by zero
    };
  });
  
  // Add lag features
  for (let i = 1; i < processed.length; i++) {
    processed[i].PM25_lag1 = processed[i - 1].PM25;
    processed[i].NO2_lag1 = processed[i - 1].NO2;
  }
  
  // First row has no lag values, use the same values
  if (processed.length > 0) {
    processed[0].PM25_lag1 = processed[0].PM25;
    processed[0].NO2_lag1 = processed[0].NO2;
  }
  
  return processed;
};

/**
 * Parse CSV data from file upload
 */
export const parseCSV = (content: string): AirQualityData[] => {
  // Split by lines and get header and rows
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',');
  
  // Process each line
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const row: any = {};
    
    headers.forEach((header, i) => {
      const value = values[i]?.trim();
      const trimmedHeader = header.trim();
      
      // Try to convert to number if possible
      if (trimmedHeader !== 'Date' && value !== '' && !isNaN(Number(value))) {
        row[trimmedHeader] = Number(value);
      } else {
        row[trimmedHeader] = value;
      }
    });
    
    return row as AirQualityData;
  });
};

/**
 * Split data into training and testing sets (80/20)
 */
export const trainTestSplit = (data: ProcessedData[]): { train: ProcessedData[], test: ProcessedData[] } => {
  // Shuffle data
  const shuffled = [...data].sort(() => 0.5 - Math.random());
  
  // Split at 80%
  const splitIndex = Math.floor(shuffled.length * 0.8);
  
  return {
    train: shuffled.slice(0, splitIndex),
    test: shuffled.slice(splitIndex)
  };
};
