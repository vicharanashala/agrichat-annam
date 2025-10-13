const DEVICE_KEY = "agrichat_device_id";

export function ensureDeviceId(): string {
  if (typeof window === "undefined") {
    return "browser";
  }
  const existing = window.localStorage.getItem(DEVICE_KEY);
  if (existing) {
    return existing;
  }
  const generated = crypto.randomUUID();
  window.localStorage.setItem(DEVICE_KEY, generated);
  return generated;
}
