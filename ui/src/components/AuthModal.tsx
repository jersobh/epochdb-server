import { useEffect, useMemo, useState } from 'react';
import { fetchDatabases, type DatabasePartition } from '../lib/api';

interface AuthModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (token: string, tenant: string, namespace: string) => void;
  initialToken: string;
  initialTenant: string;
  initialNamespace: string;
}

export function AuthModal({
  open,
  onClose,
  onSave,
  initialToken,
  initialTenant,
  initialNamespace,
}: AuthModalProps) {
  const [token, setToken] = useState(initialToken);
  const [tenant, setTenant] = useState(initialTenant);
  const [namespace, setNamespace] = useState(initialNamespace);
  const [databases, setDatabases] = useState<DatabasePartition[]>([]);

  useEffect(() => {
    if (open) {
      setToken(initialToken);
      setTenant(initialTenant);
      setNamespace(initialNamespace);
      fetchDatabases()
        .then((dbs) => setDatabases(dbs))
        .catch(() => {});
    }
  }, [open, initialToken, initialTenant, initialNamespace]);

  // Unique discovered tenants
  const discoveredTenants = useMemo(() => {
    const set = new Set<string>();
    databases.forEach((d) => {
      set.add(d.tenant || 'global');
    });
    if (tenant && tenant.trim()) set.add(tenant.trim());
    return Array.from(set).sort();
  }, [databases, tenant]);

  // Discovered namespaces for current tenant
  const discoveredNamespaces = useMemo(() => {
    const currentTenant = (tenant || 'global').trim();
    const set = new Set<string>();
    databases.forEach((d) => {
      const t = d.tenant || 'global';
      if (t === currentTenant || (!d.tenant && currentTenant === 'global')) {
        const ns = d.namespace || 'default';
        if (ns && ns !== '*' && ns !== 'all') {
          set.add(ns);
        }
      }
    });
    if (!set.has('default')) set.add('default');
    return Array.from(set).sort();
  }, [databases, tenant]);

  if (!open) return null;

  return (
    <div className="modal-overlay active" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: 440 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <h3 style={{ fontSize: 15, fontWeight: 700, fontFamily: 'var(--font-sans)', margin: 0 }}>
            Authentication & Scope
          </h3>
          <button type="button" className="close-btn" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div className="input-group">
          <label className="input-label">API Key / Token</label>
          <input
            type="password"
            className="ds-input"
            placeholder="Enter API Key..."
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
        </div>

        <div className="input-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <label className="input-label" style={{ margin: 0 }}>Tenant ID</label>
            {discoveredTenants.length > 1 && (
              <select
                style={{
                  fontSize: 11,
                  background: 'none',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 4,
                  padding: '1px 4px',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                }}
                value={tenant}
                onChange={(e) => setTenant(e.target.value)}
              >
                <option value="" disabled>Select tenant...</option>
                {discoveredTenants.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            )}
          </div>
          <input
            type="text"
            className="ds-input"
            placeholder="e.g. global"
            value={tenant}
            onChange={(e) => setTenant(e.target.value)}
            list="discovered-tenants-list"
          />
          <datalist id="discovered-tenants-list">
            {discoveredTenants.map((t) => (
              <option key={t} value={t} />
            ))}
          </datalist>
          {discoveredTenants.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6, maxHeight: 60, overflowY: 'auto' }}>
              {discoveredTenants.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTenant(t)}
                  style={{
                    fontSize: 11,
                    padding: '2px 8px',
                    borderRadius: 12,
                    border: t === tenant ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                    background: t === tenant ? 'var(--accent-teal-subtle)' : 'transparent',
                    color: t === tenant ? 'var(--accent-teal)' : 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: t === tenant ? 700 : 500,
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="input-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <label className="input-label" style={{ margin: 0 }}>Namespace</label>
            <select
              style={{
                fontSize: 11,
                background: 'none',
                border: '1px solid var(--border-subtle)',
                borderRadius: 4,
                padding: '1px 4px',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
              }}
              value={namespace}
              onChange={(e) => setNamespace(e.target.value)}
            >
              <option value="" disabled>Select namespace...</option>
              <option value="*">* (all namespaces)</option>
              {discoveredNamespaces.map((ns) => (
                <option key={ns} value={ns}>
                  {ns}
                </option>
              ))}
            </select>
          </div>
          <input
            type="text"
            className="ds-input"
            placeholder="e.g. default or *"
            value={namespace}
            onChange={(e) => setNamespace(e.target.value)}
            list="discovered-namespaces-list"
          />
          <datalist id="discovered-namespaces-list">
            <option value="*" label="* (all namespaces)" />
            {discoveredNamespaces.map((ns) => (
              <option key={ns} value={ns} />
            ))}
          </datalist>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6, maxHeight: 72, overflowY: 'auto' }}>
            <button
              type="button"
              onClick={() => setNamespace('*')}
              style={{
                fontSize: 11,
                padding: '2px 8px',
                borderRadius: 12,
                border: namespace === '*' || namespace === 'all' ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                background: namespace === '*' || namespace === 'all' ? 'var(--accent-teal-subtle)' : 'transparent',
                color: namespace === '*' || namespace === 'all' ? 'var(--accent-teal)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
                fontWeight: namespace === '*' || namespace === 'all' ? 700 : 500,
              }}
            >
              * (all)
            </button>
            {discoveredNamespaces.map((ns) => (
              <button
                key={ns}
                type="button"
                onClick={() => setNamespace(ns)}
                style={{
                  fontSize: 11,
                  padding: '2px 8px',
                  borderRadius: 12,
                  border: namespace === ns ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                  background: namespace === ns ? 'var(--accent-teal-subtle)' : 'transparent',
                  color: namespace === ns ? 'var(--accent-teal)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: namespace === ns ? 700 : 500,
                }}
              >
                {ns}
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button type="button" className="btn-ds btn-ds-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-ds btn-ds-primary"
            onClick={() => onSave(token.trim(), tenant.trim() || 'global', namespace.trim() || 'default')}
          >
            Save & Reconnect
          </button>
        </div>
      </div>
    </div>
  );
}
