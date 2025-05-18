
import React, { useState } from "react";
import { AirQualityData, ProcessedData, ModelResults } from "@/types";
import DataUploader from "@/components/DataUploader";
import DataPreprocessing from "@/components/DataPreprocessing";
import EDA from "@/components/EDA";
import FeatureEngineering from "@/components/FeatureEngineering";
import ModelTraining from "@/components/ModelTraining";
import Predictions from "@/components/Predictions";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChartBar, FileText, ChartLine } from "lucide-react";

const Index = () => {
  const [rawData, setRawData] = useState<AirQualityData[]>([]);
  const [processedData, setProcessedData] = useState<ProcessedData[]>([]);
  const [randomForestResults, setRandomForestResults] = useState<ModelResults | null>(null);
  const [xgboostResults, setXgboostResults] = useState<ModelResults | null>(null);
  const [activeTab, setActiveTab] = useState("upload");

  // Handle data upload
  const handleDataLoaded = (data: AirQualityData[]) => {
    setRawData(data);
    setActiveTab("preprocessing");
  };

  // Handle data preprocessing completion
  const handlePreprocessingComplete = (data: ProcessedData[]) => {
    setProcessedData(data);
    setActiveTab("eda");
  };

  // Handle models trained
  const handleModelsTrained = (rfResults: ModelResults, xgbResults: ModelResults) => {
    setRandomForestResults(rfResults);
    setXgboostResults(xgbResults);
    setActiveTab("predictions");
  };

  return (
    <div className="min-h-screen pb-12 bg-gradient-to-b from-background to-gray-50">
      <header className="bg-primary text-white py-6">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">AQI Analyst</h1>
              <p className="mt-1 text-primary-foreground/80">
                Advanced Air Quality Index Analysis & Prediction
              </p>
            </div>
            <div className="mt-4 md:mt-0 flex items-center space-x-2 text-sm">
              <ChartBar className="h-5 w-5" />
              <span>Data Science Dashboard</span>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 -mt-6">
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid grid-cols-2 md:grid-cols-6 mb-8">
              <TabsTrigger value="upload" disabled={false}>
                <FileText className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Upload</span>
              </TabsTrigger>
              <TabsTrigger value="preprocessing" disabled={rawData.length === 0}>
                <ChartLine className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Preprocess</span>
              </TabsTrigger>
              <TabsTrigger value="eda" disabled={processedData.length === 0}>
                <ChartBar className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">EDA</span>
              </TabsTrigger>
              <TabsTrigger value="features" disabled={processedData.length === 0}>
                <FileText className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Features</span>
              </TabsTrigger>
              <TabsTrigger value="training" disabled={processedData.length === 0}>
                <ChartLine className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Model</span>
              </TabsTrigger>
              <TabsTrigger value="predictions" disabled={!randomForestResults || !xgboostResults}>
                <ChartBar className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Results</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Data Upload</h2>
              <DataUploader onDataLoaded={handleDataLoaded} />
            </TabsContent>

            <TabsContent value="preprocessing" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Data Preprocessing</h2>
              {rawData.length > 0 && (
                <DataPreprocessing data={rawData} onComplete={handlePreprocessingComplete} />
              )}
            </TabsContent>

            <TabsContent value="eda" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Exploratory Data Analysis</h2>
              {processedData.length > 0 && <EDA data={processedData} />}
            </TabsContent>

            <TabsContent value="features" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Feature Engineering</h2>
              {processedData.length > 0 && (
                <FeatureEngineering originalData={rawData} processedData={processedData} />
              )}
            </TabsContent>

            <TabsContent value="training" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Model Training</h2>
              {processedData.length > 0 && (
                <ModelTraining
                  data={processedData}
                  onModelsTrained={handleModelsTrained}
                />
              )}
            </TabsContent>

            <TabsContent value="predictions" className="space-y-4">
              <h2 className="text-2xl font-semibold mb-4">Prediction Results</h2>
              {randomForestResults && xgboostResults && (
                <Predictions
                  randomForestResults={randomForestResults}
                  xgboostResults={xgboostResults}
                />
              )}
            </TabsContent>
          </Tabs>
        </div>

        <footer className="text-center text-gray-500 text-sm mt-12">
          <p>AQI Analyst - Advanced Air Quality Analysis Tool</p>
          <p className="mt-1">
            Built with React, TypeScript, and machine learning visualization tools
          </p>
        </footer>
      </main>
    </div>
  );
};

export default Index;
