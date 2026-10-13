import { useEffect, useState } from 'react';
import { exportData, importData } from '../lib/api';
import type { ExportBundle } from '../types';

const CATS = ['skills', 'triplets', 'profiles', 'memories'] as const;
type Cat = (typeof CATS)[number];

interface ImportExportModalProps {
  open: boolean;
  onClose: () => void;
  onToast: (msg: string) => void;
  onImported: () => void;
}

export function ImportExportModal({ open, onClose, onToast, onImported }: ImportExportModalProps) {
  const [tab, setTab] = useState<'export' | 'import'>('export');
  const [exportSel, setExportSel] = useState<Record<Cat, boolean>>({
    skills: true,
    triplets: true,
    profiles: true,
    memories: true,
  });
  const [importSel, setImportSel] = useState<Record<Cat, boolean>>({
    skills: false,
    triplets: false,
    profiles: false,
    memories: false,
  });
  const [counts, setCounts] = useState<Record<Cat, number>>({
    skills: 0,
    triplets: 0,
    profiles: 0,
    memories: 0,
  });
  const [importCounts, setImportCounts] = useState<Record<Cat, number>>({
    skills: 0,
    triplets: 0,
    profiles: 0,
    memories: 0,
  });
  const [status, setStatus] = useState('Ready');
  const [exportCache, setExportCache] = useState<ExportBundle | null>(null);
  const [stagedBundle, setStagedBundle] = useState<ExportBundle | null>(null);
  const [fileLabel, setFileLabel] = useState('Choose JSON file or drag here');
  const [busy, setBusy] = useState(false);

  const selectedExport = CATS.filter((c) => exportSel[c]);
  const selectedImport = CATS.filter((c) => importSel[c]);

  const fetchBundle = async (force = false) => {
    if (!exportCache || force) {
      const b = await exportData('all');
      setExportCache(b);
      return b;
    }
    return exportCache;
  };

  const refreshExportSummary = async (force = false) => {
    setStatus('Extracting...');
    try {
      const b = await fetchBundle(force);
      const next: Record<Cat, number> = { skills: 0, triplets: 0, profiles: 0, memories: 0 };
      let total = 0;
      CATS.forEach((c) => {
        const n = b.counts?.[c] || 0;
        next[c] = n;
        if (exportSel[c]) total += n;
      });
      setCounts(next);
      setStatus('Ready');
      return total;
    } catch {
      setStatus('Error');
      return 0;
    }
  };

  useEffect(() => {
    if (open) {
      setTab('export');
      setExportCache(null);
      void refreshExportSummary(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (open && tab === 'export' && exportCache) {
      void refreshExportSummary(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exportSel]);

  if (!open) return null;

  const exportTotal = selectedExport.reduce((a, c) => a + (counts[c] || 0), 0);

  const executeExport = async () => {
    if (!selectedExport.length) {
      onToast('Select at least one category');
      return;
    }
    setBusy(true);
    try {
      const full = await fetchBundle(true);
      const data: Record<string, unknown[]> = {};
      const nextCounts: Record<string, number> = {};
      CATS.forEach((c) => {
        const keep = selectedExport.includes(c);
        data[c] = keep ? ((full.data?.[c] as unknown[]) || []) : [];
        nextCounts[c] = data[c].length;
      });
      const bundle = {
        ...full,
        data_type: selectedExport.length === CATS.length ? 'all' : selectedExport.join('+'),
        counts: nextCounts,
        data,
      };
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `epochdb-${bundle.data_type}-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      onToast(`Exported ${selectedExport.join(', ')}`);
      onClose();
    } catch (err) {
      onToast(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const onFile = (f: File | null) => {
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const parsed = JSON.parse(String(evt.target?.result || '')) as ExportBundle;
        setStagedBundle(parsed);
        setFileLabel(f.name);
        const src = parsed.data || (parsed as unknown as Record<string, unknown[]>);
        const nextCounts: Record<Cat, number> = { skills: 0, triplets: 0, profiles: 0, memories: 0 };
        const nextSel: Record<Cat, boolean> = { skills: false, triplets: false, profiles: false, memories: false };
        CATS.forEach((c) => {
          const n = ((src as Record<string, unknown[]>)[c] || []).length;
          nextCounts[c] = n;
          nextSel[c] = n > 0;
        });
        setImportCounts(nextCounts);
        setImportSel(nextSel);
      } catch {
        onToast('Invalid JSON file');
        setStagedBundle(null);
      }
    };
    reader.readAsText(f);
  };

  const executeImport = async () => {
    if (!stagedBundle || !selectedImport.length) {
      onToast('Select at least one category');
      return;
    }
    setBusy(true);
    try {
      const src = (stagedBundle.data || stagedBundle) as Record<string, unknown[]>;
      const filtered: Record<string, unknown[]> = {};
      selectedImport.forEach((c) => {
        if (src[c]) filtered[c] = src[c];
      });
      const res = await importData({ data: filtered }, 'all') as { total_imported?: number };
      onToast(`Imported ${res.total_imported ?? 0} items!`);
      onClose();
      onImported();
    } catch (err) {
      onToast(`Import error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay active" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: 620 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-sans)' }}>
              Granular Data Migration
            </h3>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
              Export or import procedural skills, profiles, and triplets across instances
            </p>
          </div>
          <button type="button" className="close-btn" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="toggle-group" style={{ alignSelf: 'flex-start' }}>
          <button
            type="button"
            className={`toggle-btn${tab === 'export' ? ' active' : ''}`}
            onClick={() => setTab('export')}
          >
            Export Data
          </button>
          <button
            type="button"
            className={`toggle-btn${tab === 'import' ? ' active' : ''}`}
            onClick={() => setTab('import')}
          >
            Import Data
          </button>
        </div>

        {tab === 'export' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="input-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="input-label">Select Data Categories to Export</label>
                <button
                  type="button"
                  className="btn-ds btn-ds-secondary"
                  style={{ padding: '2px 8px', fontSize: 11 }}
                  onClick={() => {
                    const all = selectedExport.length === CATS.length;
                    const next = { ...exportSel };
                    CATS.forEach((c) => {
                      next[c] = !all;
                    });
                    setExportSel(next);
                  }}
                >
                  Toggle all
                </button>
              </div>
              <div className="cat-grid">
                {CATS.map((c) => (
                  <label key={c} className="cat-chip">
                    <input
                      type="checkbox"
                      checked={exportSel[c]}
                      onChange={(e) => setExportSel({ ...exportSel, [c]: e.target.checked })}
                    />
                    <span style={{ textTransform: 'capitalize' }}>{c}</span>
                    <span className="cat-count">{counts[c]}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-title">
                <span>Bundle Summary</span>
                <span style={{ fontSize: 11, color: 'var(--accent-teal)' }}>{status}</span>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                {selectedExport.length
                  ? `${exportTotal} items across ${selectedExport.length} categor${selectedExport.length === 1 ? 'y' : 'ies'} (${selectedExport.join(', ')})`
                  : 'Select at least one category.'}
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button type="button" className="btn-ds btn-ds-secondary" onClick={onClose}>
                Cancel
              </button>
              <button
                type="button"
                className="btn-ds btn-ds-primary"
                disabled={busy || selectedExport.length === 0}
                onClick={executeExport}
              >
                Download JSON Bundle
              </button>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="input-group">
              <label className="input-label">Choose or Drop JSON Bundle</label>
              <div
                style={{
                  border: '2px dashed var(--border-medium)',
                  borderRadius: 'var(--radius-md)',
                  padding: 20,
                  textAlign: 'center',
                  cursor: 'pointer',
                  background: 'var(--bg-app)',
                }}
                onClick={() => document.getElementById('ie-file-input')?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  onFile(e.dataTransfer.files?.[0] || null);
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 600 }}>{fileLabel}</div>
                <input
                  id="ie-file-input"
                  type="file"
                  accept=".json"
                  style={{ display: 'none' }}
                  onChange={(e) => onFile(e.target.files?.[0] || null)}
                />
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Select Categories to Ingest</label>
              <div className="cat-grid">
                {CATS.map((c) => (
                  <label
                    key={c}
                    className={`cat-chip${importCounts[c] === 0 ? ' disabled' : ''}`}
                  >
                    <input
                      type="checkbox"
                      checked={importSel[c]}
                      disabled={importCounts[c] === 0}
                      onChange={(e) => setImportSel({ ...importSel, [c]: e.target.checked })}
                    />
                    <span style={{ textTransform: 'capitalize' }}>{c}</span>
                    <span className="cat-count">{importCounts[c]}</span>
                  </label>
                ))}
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button type="button" className="btn-ds btn-ds-secondary" onClick={onClose}>
                Cancel
              </button>
              <button
                type="button"
                className="btn-ds btn-ds-primary"
                disabled={busy || !stagedBundle || selectedImport.length === 0}
                onClick={executeImport}
              >
                Ingest Selected
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
