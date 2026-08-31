import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const dir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(dir, "src") },
  },
  root: dir,
  publicDir: false,
  build: {
    outDir: path.resolve(dir, "../src/mlx_edge/web"),
    emptyOutDir: true,
    sourcemap: false,
    assetsDir: "assets",
  },
});
