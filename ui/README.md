# EpochDB Studio (UI)

React + TypeScript + Vite front-end for the EpochDB knowledge graph explorer.

## Development

```bash
npm install
npm run dev
```

Vite proxies API routes to `http://localhost:8080`. Start the EpochDB server (or `docker compose up`) separately.

## Production build

```bash
npm run build
```

Output is written to `../src/static/ui/` and served by FastAPI at `/` and `/visualize`. Docker builds this stage automatically.
