
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { FileX } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-background to-gray-50">
      <div className="text-center max-w-md mx-auto p-8 bg-white rounded-lg shadow-md">
        <div className="bg-gray-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
          <FileX className="h-10 w-10 text-gray-500" />
        </div>
        <h1 className="text-4xl font-bold text-primary mb-2">404</h1>
        <p className="text-xl text-gray-600 mb-6">Page Not Found</p>
        <p className="text-gray-500 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Button asChild className="w-full">
          <a href="/">Return to Home</a>
        </Button>
      </div>
    </div>
  );
};

export default NotFound;
