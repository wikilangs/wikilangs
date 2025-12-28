import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  site: 'https://wikilangs.org',
  integrations: [mdx(), sitemap()],
  alias: {
    '@lib': path.resolve(__dirname, './src/lib'),
    '@components': path.resolve(__dirname, './src/components'),
    '@layouts': path.resolve(__dirname, './src/layouts'),
    '@styles': path.resolve(__dirname, './src/styles'),
  },
  build: {
    format: 'directory'
  },
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
      wrap: true
    }
  },
  vite: {
    optimizeDeps: {
      exclude: ['fsevents']
    }
  }
});
