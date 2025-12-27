# Wikilangs Website

The official website for Wikilangs, built with [Astro](https://astro.build).

## Development

### Prerequisites

- Node.js 20+
- Python 3.11+ (for data fetching)
- `wikilangs` package installed (`pip install -e ..`)

### Local Development

```bash
# Install dependencies
npm install

# Start dev server (uses sample data from src/data/languages.json)
npm run dev

# Fetch real language data and build
npm run build

# Preview production build
npm run preview
```

### Project Structure

```
website/
├── src/
│   ├── pages/           # Astro pages (file-based routing)
│   │   ├── index.astro  # Home page
│   │   ├── languages/   # Language catalog and individual pages
│   │   ├── quickstart/  # Getting started guide
│   │   ├── docs/        # API documentation
│   │   ├── research/    # Metrics and methodology
│   │   └── about/       # About page
│   ├── layouts/         # Page layouts
│   ├── components/      # Reusable components
│   ├── lib/             # TypeScript utilities
│   ├── styles/          # Global CSS
│   └── data/            # Language data (generated at build)
├── public/              # Static assets
├── scripts/             # Build scripts
└── astro.config.mjs     # Astro configuration
```

## Build Process

The build process:

1. **Fetch Language Data** (`scripts/fetch_languages.py`)
   - Calls `wikilangs.languages_with_metadata('latest')`
   - Fetches model cards from HuggingFace for each language
   - Parses YAML frontmatter for metrics
   - Outputs `src/data/languages.json`

2. **Generate Static Pages** (`astro build`)
   - Generates 340+ language pages from template
   - References visualizations from HuggingFace CDN
   - Outputs to `dist/`

## Deployment

The site is deployed automatically via GitHub Actions:

- **Trigger**: Push to `main`, weekly schedule, or manual dispatch
- **Build**: Fetches data, builds Astro site
- **Deploy**: GitHub Pages

See `.github/workflows/website.yml` for details.

## Credits

- **Author**: [Omar Kamali](https://omarkamali.com)
- **Affiliation**: [Omneity Labs](https://omneitylabs.com)
- **Sponsor**: [Featherless.ai](https://featherless.ai)
