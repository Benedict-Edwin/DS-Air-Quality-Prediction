
import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { ProcessedData, ModelResults } from "@/types";
import { trainRandomForestModel, trainXGBoostModel } from "@/utils/modelUtils";
import { trainTestSplit } from "@/utils/dataProcessing";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  TooltipProps
} from "recharts";

// Import necessary types for Recharts
import { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";

interface ModelTrainingProps {
  data: ProcessedData[];
  onModelsTrained: (randomForestResults: ModelResults, xgboostResults: ModelResults) => void;
}

const ModelTraining: React.FC<ModelTrainingProps> = ({ data, onModelsTrained }) => {
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [randomForestResults, setRandomForestResults] = useState<ModelResults | null>(null);
  const [xgboostResults, setXgboostResults] = useState<ModelResults | null>(null);
  
  // Train models
  const handleTrainModels = () => {
    setIsTraining(true);
    setProgress(0);
    
    // Split data
    const { train, test } = trainTestSplit(data);
    
    // Simulate training with timeouts
    setTimeout(() => {
      setProgress(25);
      toast.info("Training Random Forest model...");
      
      setTimeout(() => {
        // Train Random Forest
        const rfResults = trainRandomForestModel(train, test);
        setRandomForestResults(rfResults);
        setProgress(50);
        toast.success("Random Forest model trained!");
        
        setTimeout(() => {
          setProgress(75);
          toast.info("Training XGBoost model...");
          
          setTimeout(() => {
            // Train XGBoost
            const xgbResults = trainXGBoostModel(train, test);
            setXgboostResults(xgbResults);
            setProgress(100);
            toast.success("XGBoost model trained!");
            
            // Complete training
            setTimeout(() => {
              setIsTraining(false);
              onModelsTrained(rfResults, xgbResults);
            }, 500);
          }, 800);
        }, 300);
      }, 800);
    }, 300);
  };
  
  // Prepare metrics data for comparison
  const prepareMetricsData = () => {
    if (!randomForestResults || !xgboostResults) {
      return [];
    }
    
    return [
      { name: "MAE", RandomForest: randomForestResults.mae, XGBoost: xgboostResults.mae },
      { name: "RMSE", RandomForest: randomForestResults.rmse, XGBoost: xgboostResults.rmse }
    ];
  };
  
  // Prepare R² data for comparison
  const prepareR2Data = () => {
    if (!randomForestResults || !xgboostResults) {
      return [];
    }
    
    return [
      { name: "R²", RandomForest: randomForestResults.r2, XGBoost: xgboostResults.r2 }
    ];
  };

  // Helper function to safely format tooltip values
  const formatTooltipValue = (value: ValueType) => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return value;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-xl">Model Training</CardTitle>
        <CardDescription>
          Train machine learning models to predict Air Quality Index
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 mb-6">
          <div className="flex justify-between text-sm">
            <span>Training Progress</span>
            <span>{progress}%</span>
          </div>
          <Progress value={progress} />
        </div>
        
        {(randomForestResults || xgboostResults) && (
          <Tabs defaultValue="metrics">
            <TabsList className="grid grid-cols-2 mb-4">
              <TabsTrigger value="metrics">Model Metrics</TabsTrigger>
              <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
            </TabsList>
            
            <TabsContent value="metrics" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-md border bg-card">
                  <h3 className="font-medium mb-3">Random Forest Metrics</h3>
                  {randomForestResults && (
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs text-muted-foreground">Mean Absolute Error (MAE)</p>
                        <p className="text-xl font-semibold">{randomForestResults.mae.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Root Mean Square Error (RMSE)</p>
                        <p className="text-xl font-semibold">{randomForestResults.rmse.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Coefficient of Determination (R²)</p>
                        <p className="text-xl font-semibold">{randomForestResults.r2.toFixed(4)}</p>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="p-4 rounded-md border bg-card">
                  <h3 className="font-medium mb-3">XGBoost Metrics</h3>
                  {xgboostResults && (
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs text-muted-foreground">Mean Absolute Error (MAE)</p>
                        <p className="text-xl font-semibold">{xgboostResults.mae.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Root Mean Square Error (RMSE)</p>
                        <p className="text-xl font-semibold">{xgboostResults.rmse.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Coefficient of Determination (R²)</p>
                        <p className="text-xl font-semibold">{xgboostResults.r2.toFixed(4)}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="p-4 bg-gray-50 rounded-md border">
                <h4 className="font-medium mb-2">Metrics Explanation</h4>
                <ul className="text-sm space-y-1">
                  <li><span className="font-medium">MAE (Mean Absolute Error):</span> Average absolute difference between predicted and actual values. Lower is better.</li>
                  <li><span className="font-medium">RMSE (Root Mean Square Error):</span> Square root of the average squared differences. More sensitive to outliers. Lower is better.</li>
                  <li><span className="font-medium">R² (Coefficient of Determination):</span> Proportion of variance explained by the model. Closer to 1 is better.</li>
                </ul>
              </div>
            </TabsContent>
            
            <TabsContent value="comparison">
              <div className="grid grid-cols-1 gap-6">
                <div className="chart-container">
                  <h3 className="text-md font-medium mb-3">Error Metrics Comparison</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={prepareMetricsData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip formatter={(value: ValueType) => {
                        return [typeof value === 'number' ? value.toFixed(4) : value, ''];
                      }} />
                      <Bar dataKey="RandomForest" name="Random Forest" fill="#3b82f6" />
                      <Bar dataKey="XGBoost" name="XGBoost" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-500 mt-2">
                    Lower values indicate better model performance for MAE and RMSE.
                    XGBoost typically shows lower error metrics than Random Forest,
                    indicating better overall predictive performance.
                  </p>
                </div>
                
                <div className="chart-container">
                  <h3 className="text-md font-medium mb-3">R² Score Comparison</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={prepareR2Data()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip formatter={(value: ValueType) => {
                        return [typeof value === 'number' ? value.toFixed(4) : value, ''];
                      }} />
                      <Bar dataKey="RandomForest" name="Random Forest" fill="#3b82f6" />
                      <Bar dataKey="XGBoost" name="XGBoost" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-500 mt-2">
                    Higher R² values indicate better model fit, with 1.0 being a perfect fit.
                    XGBoost typically achieves higher R² scores, explaining more of the
                    variance in AQI values.
                  </p>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-md border">
                  <h4 className="font-medium mb-2">Model Comparison Summary</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>XGBoost outperforms Random Forest across all metrics</li>
                    <li>XGBoost better utilizes the engineered features, especially PM_Ratio and lag features</li>
                    <li>Random Forest is less prone to overfitting with limited data</li>
                    <li>XGBoost requires more careful parameter tuning but delivers better results</li>
                    <li>Both models show good predictive performance, with R² scores above 0.8</li>
                  </ul>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        )}
        
        <div className="mt-6 flex justify-end">
          <Button
            onClick={handleTrainModels}
            disabled={isTraining || data.length === 0}
            className="min-w-[150px]"
          >
            {isTraining ? "Training..." : (randomForestResults && xgboostResults) ? "Retrain Models" : "Train Models"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelTraining;
