import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Same-origin /api in dev → Flask (avoids CORS; set PORT if your backend differs)
      "/api": {
        target: "http://127.0.0.1:5003",
        changeOrigin: true,
      },
    },
  },
});
