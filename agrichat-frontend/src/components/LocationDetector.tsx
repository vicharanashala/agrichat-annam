import { useState } from "react";
import { MapPin, AlertCircle, Check, X } from "lucide-react";
import { Button } from "./ui/Button";
import { getCurrentLocation } from "../utils/geolocation";

interface LocationResult {
  latitude: number;
  longitude: number;
  accuracy: number;
  state: string;
}

interface LocationDetectorProps {
  onLocationDetected: (state: string) => void;
  currentState: string;
}

export function LocationDetector({ onLocationDetected, currentState }: LocationDetectorProps) {
  const [isDetecting, setIsDetecting] = useState(false);
  const [lastDetection, setLastDetection] = useState<LocationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDetectLocation = async () => {
    setIsDetecting(true);
    setError(null);

    try {
      const result = await getCurrentLocation();
      setLastDetection(result);

      if (result.state) {
        onLocationDetected(result.state);
        setError(null);
      } else {
        setError("Could not determine state from location. Please select manually.");
      }
    } catch (err) {
      setError((err as Error).message);
      setLastDetection(null);
    } finally {
      setIsDetecting(false);
    }
  };

  const formatAccuracy = (accuracy: number): string => {
    if (accuracy < 1000) {
      return `${Math.round(accuracy)}m`;
    }
    return `${(accuracy / 1000).toFixed(1)}km`;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleDetectLocation}
          disabled={isDetecting}
          className="flex items-center gap-2"
        >
          {isDetecting ? (
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          ) : (
            <MapPin className="h-4 w-4" />
          )}
          {isDetecting ? 'Detecting...' : 'Auto-detect State'}
        </Button>

        {lastDetection && !isDetecting && (
          <div className="flex items-center gap-1 text-sm text-green-600 dark:text-green-400">
            <Check className="h-4 w-4" />
            <span>Located</span>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-red-700 dark:text-red-300">
            <p className="font-medium">Location Detection Failed</p>
            <p>{error}</p>
            {error.includes('denied') && (
              <p className="mt-1 text-xs">
                Enable location permissions in your browser to use this feature.
              </p>
            )}
          </div>
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {lastDetection && lastDetection.state && (
        <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <div className="flex items-center gap-2 text-sm">
            <MapPin className="h-4 w-4 text-green-600 dark:text-green-400" />
            <div className="flex-1">
              <p className="font-medium text-green-800 dark:text-green-200">
                Detected: {lastDetection.state}
              </p>
              <p className="text-xs text-green-600 dark:text-green-400">
                Accuracy: {formatAccuracy(lastDetection.accuracy)} • 
                Location: {lastDetection.latitude.toFixed(4)}, {lastDetection.longitude.toFixed(4)}
              </p>
            </div>
          </div>
          {lastDetection.state !== currentState && (
            <p className="text-xs text-green-600 dark:text-green-400 mt-2">
              State has been automatically updated to {lastDetection.state}
            </p>
          )}
        </div>
      )}

      <p className="text-xs text-gray-500 dark:text-gray-400">
        Auto-detection helps provide location-specific agricultural advice for your region.
      </p>
    </div>
  );
}