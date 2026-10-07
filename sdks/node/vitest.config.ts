import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: true,
    coverage: {
      provider: "v8",
      reporter: ["text", "json-summary"],
      include: ["src/**/*.ts"],
      exclude: [
        "src/**/*.spec.ts",
        "src/**/*.test.ts",
        "src/**/__tests__/**",
        "src/index.ts",
        "src/types.ts",
        "src/adapters/**",
      ],
      thresholds: {
        lines: 74,
        functions: 85,
        statements: 74,
        branches: 69,
      },
    },
  },
});
