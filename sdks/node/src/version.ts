/**
 * Central SDK identity — wire headers and observation sdk_version.
 */

export const SDK_VERSION = "tensorcost-node/1.3.0";

/** Advertise only capabilities this build implements. */
export const SDK_CAPABILITIES = ["streaming", "refusal-v1"] as const;
export type SdkCapability = (typeof SDK_CAPABILITIES)[number];

export function sdkCapabilityHeader(): string {
  return SDK_CAPABILITIES.join(",");
}
