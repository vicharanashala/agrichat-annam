interface LocationResult {
  latitude: number;
  longitude: number;
  accuracy: number;
  state: string;
}

interface NominatimResponse {
  address: {
    state?: string;
    state_district?: string;
    country?: string;
  };
  display_name: string;
}

export async function detectStateFromCoordinates(latitude: number, longitude: number): Promise<string | null> {
  try {
    // Using OpenStreetMap's Nominatim API (free, no API key required)
    const response = await fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json&addressdetails=1`,
      {
        headers: {
          'User-Agent': 'AgriChat-App/1.0' // Required by Nominatim
        }
      }
    );

    if (!response.ok) {
      throw new Error('Failed to fetch location data');
    }

    const data: NominatimResponse = await response.json();
    
    // Extract state from the address
    const state = data.address?.state;
    
    if (state) {
      return state;
    }

    return null;
  } catch (error) {
    console.error('Error detecting state from coordinates:', error);
    return null;
  }
}

export function getCurrentLocation(): Promise<LocationResult> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation is not supported by this browser'));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude, accuracy } = position.coords;
        
        try {
          const detectedState = await detectStateFromCoordinates(latitude, longitude);
          
          resolve({
            latitude,
            longitude,
            accuracy,
            state: detectedState || 'Unknown'
          });
        } catch (error) {
          reject(error);
        }
      },
      (error) => {
        let errorMessage = 'Unable to retrieve location';
        
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = 'Location access denied by user';
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage = 'Location information is unavailable';
            break;
          case error.TIMEOUT:
            errorMessage = 'Location request timed out';
            break;
        }
        
        reject(new Error(errorMessage));
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000 // 5 minutes
      }
    );
  });
}