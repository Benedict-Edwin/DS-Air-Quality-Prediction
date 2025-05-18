
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ProcessedData } from "@/types";
import { calculateCorrelationMatrix } from "@/utils/modelUtils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  Cell,
  TooltipProps
} from "recharts";

// Import necessary types for Recharts
import { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";

interface EDAProps {
  data: ProcessedData[];
}

const EDA: React.FC<EDAProps> = ({ data }) => {
  const [correlationMatrix, setCorrelationMatrix] = useState<Record<string, Record<string, number>>>({});

  useEffect(() => {
    if (data.length > 0) {
      setCorrelationMatrix(calculateCorrelationMatrix(data));
    }
  }, [data]);

  // Prepare histogram data for PM2.5
  const preparePM25Histogram = () => {
    // Check if data is empty to prevent errors
    if (!data || data.length === 0) {
      return [];
    }
    
    const values = data.map(row => row.PM25).filter(val => val !== undefined && val !== null);
    
    // If no valid values, return empty array
    if (values.length === 0) {
      return [];
    }
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    
    // Return empty array if range is invalid
    if (!range || isNaN(range) || range <= 0) {
      return [];
    }
    
    // Create 10 bins
    const binCount = 10;
    const binWidth = range / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => ({
      bin: `${(min + i * binWidth).toFixed(1)}-${(min + (i + 1) * binWidth).toFixed(1)}`,
      count: 0,
      min: min + i * binWidth,
      max: min + (i + 1) * binWidth
    }));
    
    // Count values in each bin
    values.forEach(value => {
      if (value !== undefined && value !== null) {
        const binIndex = Math.min(
          binCount - 1,
          Math.floor((value - min) / binWidth)
        );
        if (bins[binIndex]) {  // Check if bin exists before incrementing count
          bins[binIndex].count++;
        }
      }
    });
    
    return bins;
  };

  // Prepare histogram data for PM10
  const preparePM10Histogram = () => {
    // Check if data is empty to prevent errors
    if (!data || data.length === 0) {
      return [];
    }
    
    const values = data.map(row => row.PM10).filter(val => val !== undefined && val !== null);
    
    // If no valid values, return empty array
    if (values.length === 0) {
      return [];
    }
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    
    // Return empty array if range is invalid
    if (!range || isNaN(range) || range <= 0) {
      return [];
    }
    
    // Create 10 bins
    const binCount = 10;
    const binWidth = range / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => ({
      bin: `${(min + i * binWidth).toFixed(1)}-${(min + (i + 1) * binWidth).toFixed(1)}`,
      count: 0,
      min: min + i * binWidth,
      max: min + (i + 1) * binWidth
    }));
    
    // Count values in each bin
    values.forEach(value => {
      if (value !== undefined && value !== null) {
        const binIndex = Math.min(
          binCount - 1,
          Math.floor((value - min) / binWidth)
        );
        if (bins[binIndex]) {  // Check if bin exists before incrementing count
          bins[binIndex].count++;
        }
      }
    });
    
    return bins;
  };

  // Prepare histogram data for NO2
  const prepareNO2Histogram = () => {
    // Check if data is empty to prevent errors
    if (!data || data.length === 0) {
      return [];
    }
    
    const values = data.map(row => row.NO2).filter(val => val !== undefined && val !== null);
    
    // If no valid values, return empty array
    if (values.length === 0) {
      return [];
    }
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    
    // Return empty array if range is invalid
    if (!range || isNaN(range) || range <= 0) {
      return [];
    }
    
    // Create 10 bins
    const binCount = 10;
    const binWidth = range / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => ({
      bin: `${(min + i * binWidth).toFixed(1)}-${(min + (i + 1) * binWidth).toFixed(1)}`,
      count: 0,
      min: min + i * binWidth,
      max: min + (i + 1) * binWidth
    }));
    
    // Count values in each bin
    values.forEach(value => {
      if (value !== undefined && value !== null) {
        const binIndex = Math.min(
          binCount - 1,
          Math.floor((value - min) / binWidth)
        );
        if (bins[binIndex]) {  // Check if bin exists before incrementing count
          bins[binIndex].count++;
        }
      }
    });
    
    return bins;
  };

  // Prepare monthly AQI data
  const prepareMonthlyAQI = () => {
    const monthlyData: Record<number, { month: number; avgAQI: number; count: number }> = {};
    
    data.forEach(row => {
      const month = row.Month;
      if (month !== undefined) {
        if (!monthlyData[month]) {
          monthlyData[month] = { month, avgAQI: 0, count: 0 };
        }
        if (row.AQI !== undefined && row.AQI !== null) {
          monthlyData[month].avgAQI += row.AQI;
          monthlyData[month].count++;
        }
      }
    });
    
    return Object.values(monthlyData).map(item => ({
      month: item.month,
      avgAQI: item.count > 0 ? item.avgAQI / item.count : 0
    }));
  };

  // Prepare correlation data for visualization
  const prepareCorrelationData = () => {
    if (Object.keys(correlationMatrix).length === 0) {
      return [];
    }
    
    const correlationData = [];
    const features = Object.keys(correlationMatrix);
    
    for (let i = 0; i < features.length; i++) {
      for (let j = 0; j < features.length; j++) {
        correlationData.push({
          x: i,
          y: j,
          feature1: features[i],
          feature2: features[j],
          correlation: correlationMatrix[features[i]][features[j]]
        });
      }
    }
    
    return correlationData;
  };

  // Prepare data for scatter plot (PM2.5 vs AQI)
  const prepareScatterData = () => {
    return data
      .filter(row => row.PM25 !== undefined && row.PM25 !== null && 
                     row.AQI !== undefined && row.AQI !== null)
      .map(row => ({
        PM25: row.PM25,
        AQI: row.AQI
      }));
  };

  // Helper function to get color based on correlation value
  const getCorrelationColor = (correlation: number) => {
    if (correlation === 1) return "#3b82f6"; // Blue for perfect correlation
    if (correlation > 0.8) return "#4ade80"; // Green for strong positive
    if (correlation > 0.4) return "#a3e635"; // Light green for moderate positive
    if (correlation > 0) return "#d9f99d"; // Yellow-green for weak positive
    if (correlation > -0.4) return "#fef08a"; // Yellow for weak negative
    if (correlation > -0.8) return "#fdba74"; // Orange for moderate negative
    return "#f87171"; // Red for strong negative
  };

  // Helper function to safely format tooltip values
  const formatTooltipValue = (value: ValueType) => {
    if (typeof value === 'number') {
      return value.toFixed(2);
    }
    return value;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-xl">Exploratory Data Analysis</CardTitle>
        <CardDescription>
          Visualize and understand patterns in your air quality data
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="histograms">
          <TabsList className="grid grid-cols-4 mb-4">
            <TabsTrigger value="histograms">Histograms</TabsTrigger>
            <TabsTrigger value="correlation">Correlation</TabsTrigger>
            <TabsTrigger value="seasonal">Seasonal</TabsTrigger>
            <TabsTrigger value="scatter">Relationships</TabsTrigger>
          </TabsList>
          
          <TabsContent value="histograms" className="space-y-6">
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">PM2.5 Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={preparePM25Histogram()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`Count: ${value}`, 'Frequency']} labelFormatter={(label) => `PM2.5 Range: ${label}`} />
                  <Bar dataKey="count" fill="#3b82f6" name="Frequency" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                PM2.5 tends to follow a right-skewed distribution with most values in the lower range,
                but with some higher outliers that represent pollution events.
              </p>
            </div>
            
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">PM10 Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={preparePM10Histogram()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`Count: ${value}`, 'Frequency']} labelFormatter={(label) => `PM10 Range: ${label}`} />
                  <Bar dataKey="count" fill="#8b5cf6" name="Frequency" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                PM10 shows a similar distribution to PM2.5 but with higher overall values.
                The ratio between PM2.5 and PM10 can indicate different pollution sources.
              </p>
            </div>
            
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">NO2 Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={prepareNO2Histogram()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`Count: ${value}`, 'Frequency']} labelFormatter={(label) => `NO2 Range: ${label}`} />
                  <Bar dataKey="count" fill="#f43f5e" name="Frequency" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                NO2 levels are typically associated with vehicular emissions and industrial activities.
                The distribution helps identify baseline levels versus pollution events.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="correlation">
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">Correlation Heatmap</h3>
              <div className="relative overflow-x-auto">
                <div className="w-full min-w-[600px] h-[600px] max-w-full mx-auto">
                  {data.length > 0 && Object.keys(correlationMatrix).length > 0 && (
                    <div className="grid grid-cols-11 gap-0 w-full h-full">
                      {/* Column headers */}
                      <div className="bg-gray-100 flex items-center justify-center"></div>
                      {Object.keys(correlationMatrix).map((feature, idx) => (
                        <div 
                          key={idx} 
                          className="bg-gray-100 flex items-center justify-center p-1 text-xs font-semibold"
                        >
                          {feature}
                        </div>
                      ))}
                      
                      {/* Rows */}
                      {Object.keys(correlationMatrix).map((feature1, rowIdx) => (
                        <React.Fragment key={rowIdx}>
                          {/* Row header */}
                          <div 
                            className="bg-gray-100 flex items-center justify-center p-1 text-xs font-semibold"
                          >
                            {feature1}
                          </div>
                          
                          {/* Cells */}
                          {Object.keys(correlationMatrix).map((feature2, colIdx) => {
                            const correlation = correlationMatrix[feature1][feature2];
                            return (
                              <div 
                                key={`${rowIdx}-${colIdx}`}
                                className="flex items-center justify-center text-xs"
                                style={{ 
                                  backgroundColor: getCorrelationColor(correlation),
                                  color: Math.abs(correlation) > 0.7 ? 'white' : 'black'
                                }}
                              >
                                {correlation.toFixed(2)}
                              </div>
                            );
                          })}
                        </React.Fragment>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <p className="text-sm text-gray-500 mt-4">
                The correlation heatmap reveals relationships between variables. PM2.5 and PM10 show strong correlation,
                as expected. AQI is most strongly influenced by PM2.5, followed by PM10 and NO2.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="seasonal">
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">Monthly Average AQI</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={prepareMonthlyAQI()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="month"
                    tickFormatter={(month) => {
                      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
                      return monthNames[month - 1];
                    }}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value: ValueType) => {
                      return [typeof value === 'number' ? value.toFixed(2) : value, 'Average AQI'];
                    }}
                    labelFormatter={(month) => {
                      const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
                      return monthNames[Number(month) - 1];
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="avgAQI" 
                    name="Average AQI" 
                    stroke="#3b82f6" 
                    strokeWidth={2} 
                    activeDot={{ r: 8 }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                AQI peaks during winter months (December-February) due to increased heating, 
                lower mixing height, and weather conditions that trap pollutants. Summer months 
                generally show better air quality with some increases due to ozone formation.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="scatter">
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">PM2.5 vs AQI Relationship</h3>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid />
                  <XAxis type="number" dataKey="PM25" name="PM2.5" />
                  <YAxis type="number" dataKey="AQI" name="AQI" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value) => [`${value}`, '']} />
                  <Legend />
                  <Scatter name="PM2.5 vs AQI" data={prepareScatterData()} fill="#8884d8">
                    {prepareScatterData().map((entry, index) => (
                      <Cell key={index} fill={entry.AQI > 100 ? '#ef4444' : entry.AQI > 50 ? '#f97316' : '#4ade80'} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                PM2.5 shows a strong positive correlation with AQI, confirming it as a major contributor to air quality degradation.
                The points are color-coded by AQI severity, showing how PM2.5 concentrations directly impact air quality categories.
              </p>
            </div>
          </TabsContent>
        </Tabs>
        
        <div className="mt-6 bg-gray-50 p-4 rounded-md border border-gray-200">
          <h3 className="font-medium mb-2">Key Insights</h3>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>PM2.5 is the strongest contributor to AQI, with correlation coefficient around 0.9</li>
            <li>AQI values show distinct seasonal patterns with peaks during winter months</li>
            <li>PM2.5 and PM10 are highly correlated (approx. 0.8), suggesting common sources</li>
            <li>Temperature shows negative correlation with pollutants, as warm air helps dispersion</li>
            <li>NO2 peaks during morning and evening rush hours, indicating traffic as a major source</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default EDA;
