import type { ViewMode } from '../types';

interface TopNavProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  onToggleDock: () => void;
  dockOpen: boolean;
  onOpenImportExport: () => void;
  onOpenAddTriplet: () => void;
  onToggleTheme: () => void;
  onOpenAuth: () => void;
  tenant: string;
  namespace: string;
  isConnected: boolean;
  nodeCount: number;
  edgeCount: number;
  darkMode: boolean;
}

export function TopNav({
  viewMode,
  onViewModeChange,
  onToggleDock,
  dockOpen,
  onOpenImportExport,
  onOpenAddTriplet,
  onToggleTheme,
  onOpenAuth,
  tenant,
  namespace,
  isConnected,
  nodeCount,
  edgeCount,
  darkMode,
}: TopNavProps) {
  return (
    <header className="top-nav">
      <div className="brand-section">
        <div className="brand-monogram">ED</div>
        <div className="brand-title-group">
          <h1>EpochDB</h1>
          <p>
            <span className={`pulse-dot${isConnected ? '' : ' offline'}`} />
            <span>{isConnected ? 'Cluster Online' : 'Connecting...'}</span>
            <span style={{ opacity: 0.4 }}>·</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>
              {nodeCount} nodes · {edgeCount} triples
            </span>
          </p>
        </div>

        <div className="scope-badge" onClick={onOpenAuth} title="Configure Tenant / Namespace / API Key">
          <span>scope:</span>
          <b>{tenant}</b>
          <span>/</span>
          <b>{namespace}</b>
        </div>
      </div>

      <div className="header-actions">
        <div className="toggle-group">
          <button
            type="button"
            className={`toggle-btn${viewMode === '2d' ? ' active' : ''}`}
            onClick={() => onViewModeChange('2d')}
          >
            2D
          </button>
          <button
            type="button"
            className={`toggle-btn${viewMode === '3d' ? ' active' : ''}`}
            onClick={() => onViewModeChange('3d')}
          >
            3D
          </button>
        </div>

        <button
          type="button"
          className={`btn-ds btn-ds-secondary${dockOpen ? ' is-on' : ''}`}
          onClick={onToggleDock}
          title="Show / hide Explorer & Telemetry"
        >
          <svg className="icon-svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 3v18" />
          </svg>
          <span>Explorer</span>
        </button>

        <button type="button" className="btn-ds btn-ds-secondary" onClick={onOpenImportExport} title="Granular Import / Export">
          <span>Import / Export</span>
        </button>

        <button type="button" className="btn-ds btn-ds-primary" onClick={onOpenAddTriplet}>
          <span>+ Add Triplet</span>
        </button>

        <button
          type="button"
          className="btn-ds btn-ds-secondary"
          style={{ padding: '7px 10px' }}
          onClick={onToggleTheme}
          title="Switch Light / Dark Theme"
          aria-label="Toggle theme"
        >
          {darkMode ? (
            <svg className="icon-svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
            </svg>
          ) : (
            <svg className="icon-svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>

        <button
          type="button"
          className="btn-ds btn-ds-secondary"
          style={{ padding: '7px 10px' }}
          onClick={onOpenAuth}
          title="Authentication Key"
        >
          <svg className="icon-svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="8" cy="15" r="4" />
            <path d="M10.85 12.15L19 4M18 5l2 2M15 8l2 2" />
          </svg>
        </button>
      </div>
    </header>
  );
}
