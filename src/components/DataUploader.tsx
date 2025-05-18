
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { Upload } from "lucide-react";
import { parseCSV } from "@/utils/dataProcessing";
import { AirQualityData } from "@/types";

interface DataUploaderProps {
  onDataLoaded: (data: AirQualityData[]) => void;
}

const DataUploader: React.FC<DataUploaderProps> = ({ onDataLoaded }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState("");
  
  // Handle file upload
  const handleFileUpload = (file: File) => {
    setIsUploading(true);
    setFileName(file.name);
    
    const reader = new FileReader();
    
    reader.onload = (event) => {
      try {
        const content = event.target?.result as string;
        const data = parseCSV(content);
        
        if (data.length === 0) {
          toast.error("No data found in the file");
          return;
        }
        
        toast.success(`Successfully loaded ${data.length} rows of data`);
        onDataLoaded(data);
      } catch (error) {
        console.error("Error parsing CSV:", error);
        toast.error("Failed to parse the CSV file");
      } finally {
        setIsUploading(false);
      }
    };
    
    reader.onerror = () => {
      toast.error("Error reading file");
      setIsUploading(false);
    };
    
    reader.readAsText(file);
  };
  
  // Handle drag and drop events
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };
  
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };
  
  const loadSampleData = () => {
    setIsUploading(true);
    setFileName("synthetic_air_quality_data.csv");
    
    // This is simulating loading sample data
    // In a real app, we'd fetch this from a server
    setTimeout(() => {
      // Generate some synthetic data
      const sampleData: AirQualityData[] = Array.from({ length: 500 }, (_, i) => {
        const date = new Date();
        date.setDate(date.getDate() - (500 - i));
        
        // Generate seasonal patterns
        const dayOfYear = Math.floor((date.getTime() - new Date(date.getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24));
        const seasonalFactor = Math.sin(dayOfYear / 365 * 2 * Math.PI); // Higher in winter
        
        // Base values with some randomness
        const pm25Base = 25 + seasonalFactor * 15;
        const pm10Base = 50 + seasonalFactor * 25;
        const no2Base = 15 + seasonalFactor * 10;
        
        // Add day-to-day variation
        const pm25 = Math.max(2, pm25Base + (Math.random() - 0.5) * 10);
        const pm10 = Math.max(5, pm10Base + (Math.random() - 0.5) * 20);
        const no2 = Math.max(1, no2Base + (Math.random() - 0.5) * 8);
        const so2 = Math.max(0.5, 5 + seasonalFactor * 3 + (Math.random() - 0.5) * 4);
        const co = Math.max(0.2, 1.5 + seasonalFactor * 1 + (Math.random() - 0.5) * 1);
        const o3 = Math.max(10, 40 - seasonalFactor * 15 + (Math.random() - 0.5) * 15); // Higher in summer
        
        // Environmental factors
        const temperature = Math.max(-10, 15 - seasonalFactor * 15 + (Math.random() - 0.5) * 5);
        const humidity = Math.min(100, Math.max(30, 60 + seasonalFactor * 20 + (Math.random() - 0.5) * 15));
        const windSpeed = Math.max(0, 4 + (Math.random() - 0.5) * 6);
        
        // Calculate AQI based on pollutants
        const aqi = 0.5 * pm25 + 0.2 * pm10 + 0.15 * no2 + 0.1 * o3 + 0.05 * so2 + 0.02 * co;
        
        return {
          Date: date.toISOString().split('T')[0],
          PM25: pm25,
          PM10: pm10,
          NO2: no2,
          SO2: so2,
          CO: co,
          O3: o3,
          Temperature: temperature,
          Humidity: humidity,
          WindSpeed: windSpeed,
          AQI: aqi
        };
      });
      
      toast.success(`Loaded ${sampleData.length} rows of sample data`);
      onDataLoaded(sampleData);
      setIsUploading(false);
    }, 800);
  };

  return (
    <Card className="w-full max-w-3xl mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl">Air Quality Data Analysis</CardTitle>
        <CardDescription>
          Upload your air quality CSV data or use our sample dataset to begin analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
            isDragging ? "border-primary bg-primary/10" : "border-gray-300 hover:border-primary/50"
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById("file-upload")?.click()}
        >
          <input
            id="file-upload"
            type="file"
            accept=".csv"
            onChange={handleFileInput}
            className="hidden"
          />
          <Upload className="h-10 w-10 mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-semibold mb-1">
            {fileName ? `Selected: ${fileName}` : "Drop your CSV file here"}
          </h3>
          <p className="text-sm text-gray-500 mb-2">
            or click to browse for your file
          </p>
          <p className="text-xs text-gray-400">
            Supports CSV files with air quality data
          </p>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={() => setFileName("")} disabled={!fileName || isUploading}>
          Clear
        </Button>
        <Button onClick={loadSampleData} disabled={isUploading}>
          {isUploading ? "Loading..." : "Use Sample Data"}
        </Button>
      </CardFooter>
    </Card>
  );
};

export default DataUploader;
