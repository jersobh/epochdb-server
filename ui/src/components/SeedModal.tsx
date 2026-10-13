import { useMemo, useState } from 'react';
import { SEED_DOMAINS } from '../lib/seed';
import { rememberSkill, rememberTriple } from '../lib/api';

interface SeedModalProps {
  open: boolean;
  onClose: () => void;
  tenant: string;
  namespace: string;
  onDone: (message: string) => void;
}

export function SeedModal({ open, onClose, tenant, namespace, onDone }: SeedModalProps) {
  const [progress, setProgress] = useState('');
  const [busy, setBusy] = useState(false);

  const total = useMemo(
    () => SEED_DOMAINS.reduce((a, d) => a + d.triples.length, 0),
    []
  );

  if (!open) return null;

  const confirmSeed = async () => {
    setBusy(true);
    const all = SEED_DOMAINS.flatMap((d) => d.triples);
    let done = 0;
    let failed = 0;
    const CONCURRENCY = 6;
    const queue = all.slice();
    const worker = async () => {
      while (queue.length) {
        const triple = queue.shift();
        if (!triple) break;
        const [s, p, o] = triple;
        try {
          await rememberTriple(s, p, o);
        } catch {
          failed++;
        }
        done++;
        setProgress(`Seeding ${done} / ${all.length}...`);
      }
    };
    await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
    try {
      await rememberSkill({
        skill_name: 'Restart Payment Service',
        description:
          'Safely restart the Payment Service on the Kubernetes Cluster without dropping in-flight transactions.',
        steps: [
          { step: 1, action: 'Drain traffic from Payment Service pods' },
          { step: 2, action: 'Perform a rolling restart of the deployment' },
          { step: 3, action: 'Verify health checks and error rate before restoring traffic' },
        ],
      });
    } catch {
      /* optional */
    }
    setBusy(false);
    setProgress('');
    onClose();
    onDone(failed ? `Seeded with ${failed} failures` : `Seeded ${all.length} facts`);
  };

  return (
    <div className="modal-overlay active" onClick={busy ? undefined : onClose}>
      <div className="modal-card" style={{ maxWidth: 480 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-sans)' }}>
            Seed Sample Knowledge?
          </h3>
          <button type="button" className="close-btn" onClick={onClose} disabled={busy}>
            &times;
          </button>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          This will write the following sample facts into scope{' '}
          <b style={{ color: 'var(--accent-teal)' }}>
            {tenant} / {namespace}
          </b>
          . Domains are interlinked so you can explore multi-hop reasoning (e.g. Marie Curie → Warsaw →
          Poland → European Union).
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12 }}>
          {SEED_DOMAINS.map((d) => (
            <div
              key={d.name}
              className="scope-badge"
              style={{ cursor: 'default', justifyContent: 'space-between', width: '100%' }}
            >
              <span>{d.name}</span>
              <b>{d.triples.length} facts</b>
            </div>
          ))}
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            {total} facts + 1 procedural skill will be added (existing data is kept).
          </div>
        </div>
        <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', minHeight: 14 }}>
          {progress}
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="btn-ds btn-ds-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn-ds btn-ds-primary" onClick={confirmSeed} disabled={busy}>
            Seed Data
          </button>
        </div>
      </div>
    </div>
  );
}
