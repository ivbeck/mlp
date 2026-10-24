import { defineConfig } from "vite";

// `npm run dev` proxies the api to a running `cargo run -p mlp-console`
export default defineConfig({
  esbuild: { jsx: "automatic", jsxImportSource: "preact" },
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:7878", ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
  },
});
