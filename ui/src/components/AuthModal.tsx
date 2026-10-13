import { useEffect, useState } from 'react';

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

  useEffect(() => {
    if (open) {
      setToken(initialToken);
      setTenant(initialTenant);
      setNamespace(initialNamespace);
    }
  }, [open, initialToken, initialTenant, initialNamespace]);

  if (!open) return null;

  return (
    <div className="modal-overlay active" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: 440 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: 15, fontWeight: 700, fontFamily: 'var(--font-sans)' }}>
            Authentication & Scope
          </h3>
          <button type="button" className="close-btn" onClick={onClose}>
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
          <label className="input-label">Tenant ID</label>
          <input
            type="text"
            className="ds-input"
            placeholder="e.g. global"
            value={tenant}
            onChange={(e) => setTenant(e.target.value)}
          />
        </div>
        <div className="input-group">
          <label className="input-label">Namespace</label>
          <input
            type="text"
            className="ds-input"
            placeholder="e.g. default"
            value={namespace}
            onChange={(e) => setNamespace(e.target.value)}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
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
