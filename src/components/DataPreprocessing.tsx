
import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { AirQualityData, ProcessedData, DatasetStats } from "@/types";
import {
  imputeMissingValues,
  removeDuplicates,
  treatOutliers,
  normalizeData,
  engineerFeatures,
  calculateDatasetStats
} from "@/utils/dataProcessing";

interface DataPreprocessingProps {
  data: AirQualityData[];
  onComplete: (processedData: ProcessedData[]) => void;
}

const DataPreprocessing: React.FC<DataPreprocessingProps> = ({ data, onComplete }) => {
  const [currentData, setCurrentData] = useState<AirQualityData[]>(data);
  const [processedData, setProcessedData] = useState<ProcessedData[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState(0);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  
  // Calculate initial stats
  useEffect(() => {
    setStats(calculateDatasetStats(data));
  }, [data]);
  
  // Update stats when data changes during processing
  useEffect(() => {
    if (currentData !== data) {
      setStats(calculateDatasetStats(currentData));
    }
  }, [currentData, data]);
  
  const handleProcess = () => {
    setIsProcessing(true);
    setProgress(0);
    setStep(0);
    
    // Simulate processing with timeouts to show progress
    setTimeout(() => {
      // Step 1: Handle missing values
      const dataWithoutMissing = imputeMissingValues(currentData);
      setCurrentData(dataWithoutMissing);
      setProgress(20);
      setStep(1);
      toast.success("Missing values handled using median imputation");
      
      setTimeout(() => {
        // Step 2: Remove duplicates
        const uniqueData = removeDuplicates(dataWithoutMissing);
        setCurrentData(uniqueData);
        setProgress(40);
        setStep(2);
        toast.success(`Removed ${dataWithoutMissing.length - uniqueData.length} duplicate entries`);
        
        setTimeout(() => {
          // Step 3: Treat outliers
          const dataWithoutOutliers = treatOutliers(uniqueData);
          setCurrentData(dataWithoutOutliers);
          setProgress(60);
          setStep(3);
          toast.success("Outliers treated using IQR method");
          
          setTimeout(() => {
            // Step 4: Normalize data
            const normalizedData = normalizeData(dataWithoutOutliers);
            setCurrentData(normalizedData);
            setProgress(80);
            setStep(4);
            toast.success("Data normalized using Min-Max scaling");
            
            setTimeout(() => {
              // Step 5: Engineer features
              const engineeredData = engineerFeatures(normalizedData);
              setProcessedData(engineeredData);
              setProgress(100);
              setStep(5);
              toast.success("Feature engineering complete");
              
              // Complete preprocessing
              setTimeout(() => {
                setIsProcessing(false);
                onComplete(engineeredData);
              }, 500);
            }, 500);
          }, 500);
        }, 500);
      }, 500);
    }, 500);
  };
  
  // Display a preview of the data with safe handling of undefined values
  const renderDataPreview = () => {
    const previewData = currentData.slice(0, 5);
    
    return (
      <div className="overflow-x-auto mt-4">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
              <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">PM2.5</th>
              <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">PM10</th>
              <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">NO2</th>
              <th className="px-2 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">AQI</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {previewData.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="px-2 py-2 whitespace-nowrap text-sm">{row.Date}</td>
                <td className="px-2 py-2 whitespace-nowrap text-sm">{typeof row.PM25 === 'number' ? row.PM25.toFixed(2) : 'N/A'}</td>
                <td className="px-2 py-2 whitespace-nowrap text-sm">{typeof row.PM10 === 'number' ? row.PM10.toFixed(2) : 'N/A'}</td>
                <td className="px-2 py-2 whitespace-nowrap text-sm">{typeof row.NO2 === 'number' ? row.NO2.toFixed(2) : 'N/A'}</td>
                <td className="px-2 py-2 whitespace-nowrap text-sm">{typeof row.AQI === 'number' ? row.AQI.toFixed(2) : 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-gray-500 mt-2">Showing 5 of {currentData.length} rows</p>
      </div>
    );
  };
  
  // Steps in the preprocessing pipeline
  const steps = [
    "Start",
    "Handle Missing Values",
    "Remove Duplicates",
    "Treat Outliers",
    "Normalize Data",
    "Engineer Features"
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">Data Preprocessing</CardTitle>
        <CardDescription>
          Clean and prepare your data for analysis and modeling
        </CardDescription>
      </CardHeader>
      <CardContent>
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-xs text-gray-500">Rows</p>
              <p className="text-lg font-semibold">{stats.rowCount}</p>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-xs text-gray-500">Columns</p>
              <p className="text-lg font-semibold">{stats.columnCount}</p>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-xs text-gray-500">Missing Values</p>
              <p className="text-lg font-semibold">{stats.missingValueCount}</p>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-xs text-gray-500">Duplicates</p>
              <p className="text-lg font-semibold">{stats.duplicateCount}</p>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-xs text-gray-500">Outliers</p>
              <p className="text-lg font-semibold">{stats.outlierCount}</p>
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <p className="text-xs text-gray-500">Clean Rate</p>
              <p className="text-lg font-semibold">
                {Math.round(
                  (stats.rowCount - stats.duplicateCount - Math.min(stats.missingValueCount, stats.rowCount)) /
                    stats.rowCount *
                    100
                )}%
              </p>
            </div>
          </div>
        )}
        
        <Tabs defaultValue="pipeline">
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="pipeline">Pipeline Steps</TabsTrigger>
            <TabsTrigger value="preview">Data Preview</TabsTrigger>
          </TabsList>
          
          <TabsContent value="pipeline" className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm mb-1">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
            
            <div className="space-y-4">
              {steps.map((stepName, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-md flex items-center border ${
                    step >= i ? "bg-primary/10 border-primary" : "bg-gray-50 border-gray-200"
                  }`}
                >
                  <div
                    className={`h-6 w-6 rounded-full mr-3 flex items-center justify-center text-xs font-medium ${
                      step > i
                        ? "bg-primary text-white"
                        : step === i
                        ? "bg-primary/20 text-primary border-2 border-primary"
                        : "bg-gray-200 text-gray-500"
                    }`}
                  >
                    {step > i ? "✓" : i + 1}
                  </div>
                  <span className={step >= i ? "font-medium" : "text-gray-500"}>{stepName}</span>
                </div>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="preview">
            {renderDataPreview()}
          </TabsContent>
        </Tabs>
        
        <div className="mt-6 flex justify-end">
          <Button
            onClick={handleProcess}
            disabled={isProcessing}
            className="min-w-[150px]"
          >
            {isProcessing ? "Processing..." : step === 5 ? "Processed" : "Process Data"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default DataPreprocessing;
