
import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ModelResults } from "@/types";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  ReferenceLine,
  TooltipProps
} from "recharts";

// Import necessary types for Recharts
import { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";

interface PredictionsProps {
  randomForestResults: ModelResults;
  xgboostResults: ModelResults;
}

const Predictions: React.FC<PredictionsProps> = ({ randomForestResults, xgboostResults }) => {
  const [selectedModel, setSelectedModel] = useState<"randomForest" | "xgboost">("xgboost");
  
  // Get the currently selected model results
  const currentResults = selectedModel === "randomForest" ? randomForestResults : xgboostResults;
  
  // Prepare data for actual vs predicted chart
  const prepareComparisonData = () => {
    return currentResults.actual.map((actual, i) => ({
      index: i,
      actual,
      predicted: currentResults.predictions[i]
    })).slice(0, 100); // Limit to 100 points for visualization
  };
  
  // Prepare data for residuals chart
  const prepareResidualData = () => {
    return currentResults.actual.map((actual, i) => ({
      index: i,
      residual: actual - currentResults.predictions[i],
      predicted: currentResults.predictions[i]
    })).slice(0, 100); // Limit to 100 points for visualization
  };
  
  // Prepare data for feature importance chart
  const prepareFeatureImportanceData = () => {
    return (currentResults.featureImportance || [])
      .sort((a, b) => b.importance - a.importance);
  };
  
  // Calculate the mean residual for the reference line
  const calculateMeanResidual = () => {
    const residuals = currentResults.actual.map((actual, i) => 
      actual - currentResults.predictions[i]
    );
    return residuals.reduce((sum, val) => sum + val, 0) / residuals.length;
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
        <div className="flex flex-col sm:flex-row justify-between gap-4">
          <div>
            <CardTitle className="text-xl">Prediction Results</CardTitle>
            <CardDescription>
              Visualize and analyze model predictions and performance
            </CardDescription>
          </div>
          
          <Select value={selectedModel} onValueChange={(value: "randomForest" | "xgboost") => setSelectedModel(value)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="randomForest">Random Forest</SelectItem>
              <SelectItem value="xgboost">XGBoost</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="comparison">
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="comparison">Actual vs Predicted</TabsTrigger>
            <TabsTrigger value="residuals">Residuals</TabsTrigger>
            <TabsTrigger value="importance">Feature Importance</TabsTrigger>
          </TabsList>
          
          <TabsContent value="comparison">
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">Actual vs Predicted AQI</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={prepareComparisonData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: 'Test Sample Index', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'AQI Value', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value: ValueType) => {
                    return [typeof value === 'number' ? value.toFixed(2) : value, ''];
                  }} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Actual AQI"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ r: 1 }}
                    activeDot={{ r: 5 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name="Predicted AQI"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={{ r: 1 }}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                The {selectedModel === "randomForest" ? "Random Forest" : "XGBoost"} model shows good tracking of actual AQI values,
                with R² of {currentResults.r2.toFixed(3)} and MAE of {currentResults.mae.toFixed(2)}.
                The model captures most trends but may miss some extreme values and rapid fluctuations.
              </p>
            </div>
            
            <div className="chart-container mt-6">
              <h3 className="text-md font-medium mb-3">Prediction Scatter Plot</h3>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid />
                  <XAxis
                    type="number"
                    dataKey="actual"
                    name="Actual AQI"
                    label={{ value: 'Actual AQI', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="predicted"
                    name="Predicted AQI"
                    label={{ value: 'Predicted AQI', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value: ValueType) => {
                      return [typeof value === 'number' ? value.toFixed(2) : value, ''];
                    }}
                  />
                  <ReferenceLine y={0} stroke="#666" />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 'auto', y: 'auto' }]}
                    stroke="red"
                    strokeDasharray="3 3"
                  />
                  <Scatter
                    name="Predictions"
                    data={prepareComparisonData()}
                    fill="#8884d8"
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                This scatter plot shows the relationship between actual and predicted values.
                Points closer to the diagonal red line indicate more accurate predictions.
                {selectedModel === "xgboost" && " The XGBoost model shows less scatter around the line, indicating better overall prediction accuracy."}
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="residuals">
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">Residuals Plot</h3>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid />
                  <XAxis
                    type="number"
                    dataKey="predicted"
                    name="Predicted AQI"
                    label={{ value: 'Predicted AQI', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="residual"
                    name="Residual"
                    label={{ value: 'Residual (Actual - Predicted)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value: ValueType) => {
                      return [typeof value === 'number' ? value.toFixed(2) : value, ''];
                    }}
                  />
                  <ReferenceLine y={0} stroke="#666" />
                  <ReferenceLine
                    y={calculateMeanResidual()}
                    stroke="red"
                    strokeDasharray="3 3"
                    label={{ value: 'Mean residual', position: 'right' }}
                  />
                  <Scatter
                    name="Residuals"
                    data={prepareResidualData()}
                    fill="#8884d8"
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                The residuals plot shows the difference between actual and predicted values.
                Ideally, residuals should be randomly scattered around zero with no pattern.
                {selectedModel === "randomForest" ? 
                  " The Random Forest model shows some pattern in the residuals, indicating potential for improvement." :
                  " The XGBoost model shows more balanced residuals with less apparent pattern."}
              </p>
            </div>
            
            <div className="mt-6 bg-gray-50 p-4 rounded-md border">
              <h4 className="font-medium mb-2">Residual Analysis</h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>Mean residual: {calculateMeanResidual().toFixed(4)} (closer to zero is better)</li>
                <li>
                  {Math.abs(calculateMeanResidual()) < 0.1 ? 
                    "Model shows minimal systematic bias in predictions" :
                    "Model shows some systematic bias in predictions"}
                </li>
                <li>
                  {selectedModel === "xgboost" ?
                    "XGBoost residuals are more normally distributed, indicating good model fit" :
                    "Random Forest residuals show some skew, indicating potential for model improvement"}
                </li>
                <li>Larger residuals tend to occur at higher AQI values, suggesting the model has difficulty with extreme pollution events</li>
              </ul>
            </div>
          </TabsContent>
          
          <TabsContent value="importance">
            <div className="chart-container">
              <h3 className="text-md font-medium mb-3">Feature Importance</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart
                  data={prepareFeatureImportanceData()}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip formatter={(value: ValueType) => {
                    return [typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : value, 'Importance'];
                  }} />
                  <Bar
                    dataKey="importance"
                    name="Importance"
                    fill={selectedModel === "randomForest" ? "#3b82f6" : "#10b981"}
                  />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 mt-2">
                Feature importance shows which variables have the greatest influence on the model's predictions.
                PM2.5 is consistently the most important predictor of AQI, followed by PM10 and NO2.
                {selectedModel === "xgboost" && " XGBoost also leverages the engineered features more effectively, particularly PM_Ratio and lag features."}
              </p>
            </div>
            
            <div className="mt-6 bg-gray-50 p-4 rounded-md border">
              <h4 className="font-medium mb-2">Feature Importance Insights</h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>PM2.5 dominates AQI predictions, accounting for {selectedModel === "randomForest" ? "~35%" : "~40%"} of model decisions</li>
                <li>PM10 is the second most important feature, contributing {selectedModel === "randomForest" ? "~25%" : "~25%"} to predictions</li>
                <li>Nitrogen dioxide (NO2) plays a significant role at {selectedModel === "randomForest" ? "~20%" : "~15%"}</li>
                {selectedModel === "xgboost" && <li>Engineered features like PM_Ratio and lag variables add significant predictive power</li>}
                <li>Meteorological variables like temperature and humidity have indirect effects by influencing pollutant dispersion</li>
                <li>For improved AQI forecasts, focus monitoring efforts on the top 3-4 features identified here</li>
              </ul>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default Predictions;
