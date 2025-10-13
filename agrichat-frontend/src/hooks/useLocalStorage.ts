import { useEffect, useState } from "react";

export function useLocalStorage<T>(key: string, defaultValue: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") {
      return defaultValue;
    }
    try {
      const item = window.localStorage.getItem(key);
      if (item !== null) {
        try {
          return JSON.parse(item) as T;
        } catch (parseError) {
          console.warn(`Failed to parse localStorage key ${key} as JSON, falling back to raw value`, parseError);
          if (typeof defaultValue === "string") {
            return item as unknown as T;
          }
          if (typeof defaultValue === "boolean") {
            if (item === "true" || item === "false") {
              return (item === "true") as unknown as T;
            }
          }
          if (typeof defaultValue === "number") {
            const parsedNumber = Number(item);
            if (!Number.isNaN(parsedNumber)) {
              return parsedNumber as unknown as T;
            }
          }
        }
      }
    } catch (error) {
      console.warn(`Failed to read localStorage key ${key}`, error);
    }
    return defaultValue;
  });

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.warn(`Failed to write localStorage key ${key}`, error);
    }
  }, [key, value]);

  return [value, setValue];
}
