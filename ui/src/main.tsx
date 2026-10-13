import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// StrictMode intentionally omitted: vis-network / 3d-force-graph own DOM children
// and double-mounting causes removeChild NotFoundError crashes.
createRoot(document.getElementById('root')!).render(<App />)
