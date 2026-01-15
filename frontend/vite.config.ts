import { defineConfig } from 'vite';

// Dynamically import ESM-only plugins to avoid esbuild `require` errors when loading the config.
export default defineConfig(async () => {
  const { default: react } = await import('@vitejs/plugin-react');

  // Proxy /api and /ws to the backend during development
  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8001',
          changeOrigin: true,
          secure: false,
        },
        '/ws': {
          target: 'ws://localhost:8001',
          ws: true,
          changeOrigin: true,
        },
      },
    },
  };
});
