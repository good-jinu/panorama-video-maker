import type { NextConfig } from 'next';
import path from 'node:path';

// Share the repo's .env (GOOGLE_GENERATIVE_AI_API_KEY etc.) with the Python server.
try {
  process.loadEnvFile(path.join(__dirname, '..', '.env'));
} catch {
  // No .env: rely on the shell environment.
}

const studioUrl = process.env.STUDIO_API_URL ?? 'http://127.0.0.1:8000';

const nextConfig: NextConfig = {
  // Rendered videos are served by the Python server's /outputs static mount.
  async rewrites() {
    return [{ source: '/outputs/:path*', destination: `${studioUrl}/outputs/:path*` }];
  },
};

export default nextConfig;
