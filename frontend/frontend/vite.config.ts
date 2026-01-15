import { defineConfig } from 'vite';

// Use async config and dynamic import for ESM-only plugins to avoid esbuild require errors
export default defineConfig(async () => {
  const reactPlugin = (await import('@vitejs/plugin-react')).default;
  return {
    plugins: [reactPlugin()],
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8001',
          changeOrigin: true,
          secure: false,
        }
      }
    },
  };
});
