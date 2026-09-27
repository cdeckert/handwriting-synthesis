# Web app

The web interface is a React and TypeScript single-page app built with Vite. It talks to the Flask API in `../web_app.py`.

```bash
npm ci
npm run dev
```

The development server runs at <http://localhost:5173> and proxies `/api` to <http://localhost:5000>.

Available checks:

```bash
npm run lint
npm run typecheck
npm run build
npm run check
```

Production assets are generated into `dist/`. They are intentionally not committed; the repository Dockerfile creates them in its frontend build stage.
