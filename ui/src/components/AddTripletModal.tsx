import { useEffect, useState } from 'react';
import { TypeaheadInput } from './TypeaheadInput';
import { addTriplet } from '../lib/api';
import { DEFAULT_PREDICATES } from '../lib/graph';
import type { TypeaheadItem } from '../types';

interface AddTripletModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  onToast: (msg: string) => void;
  suggestEntities: (q: string) => TypeaheadItem[];
  knownPredicates: string[];
}

export function AddTripletModal({
  open,
  onClose,
  onSuccess,
  onToast,
  suggestEntities,
  knownPredicates,
}: AddTripletModalProps) {
  const [subject, setSubject] = useState('');
  const [predicate, setPredicate] = useState('');
  const [object, setObject] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open) {
      setSubject('');
      setPredicate('');
      setObject('');
    }
  }, [open]);

  if (!open) return null;

  const preds = Array.from(new Set([...DEFAULT_PREDICATES, ...knownPredicates])).slice(0, 12);

  const submit = async () => {
    const s = subject.trim();
    const p = predicate.trim();
    const o = object.trim();
    if (!s || !p || !o) {
      onToast('Please complete all triplet fields');
      return;
    }
    setBusy(true);
    try {
      await addTriplet(s, p, o);
      onToast('Triplet ingested successfully!');
      onClose();
      onSuccess();
    } catch (err) {
      onToast(`Ingest error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay active" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-sans)' }}>
            Add Knowledge Triplet
          </h3>
          <button type="button" className="close-btn" onClick={onClose}>
            &times;
          </button>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Directly inject relational triples into the distributed neuro-symbolic mesh.
        </p>

        <div className="input-group">
          <label className="input-label">Subject</label>
          <TypeaheadInput
            placeholder="e.g. Nikola Tesla, Kubernetes Pod"
            value={subject}
            onChange={setSubject}
            getItems={suggestEntities}
          />
        </div>

        <div className="input-group">
          <label className="input-label">Predicate</label>
          <TypeaheadInput
            placeholder="e.g. invented, relies_on, manages"
            value={predicate}
            onChange={setPredicate}
            getItems={(v) => {
              const ql = (v || '').trim().toLowerCase();
              return Array.from(new Set([...knownPredicates, ...DEFAULT_PREDICATES]))
                .filter((p) => !ql || p.includes(ql.replace(/\s+/g, '_')))
                .map((p) => ({
                  label: p,
                  value: p,
                  tag: knownPredicates.includes(p) ? 'in graph' : 'common',
                }));
            }}
          />
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
            {preds.map((p) => (
              <button
                key={p}
                type="button"
                className="btn-ds btn-ds-secondary"
                style={{ padding: '3px 8px', fontSize: 11 }}
                onClick={() => setPredicate(p)}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="input-group">
          <label className="input-label">Object</label>
          <TypeaheadInput
            placeholder="e.g. AC Motor, Deployment Spec"
            value={object}
            onChange={setObject}
            getItems={suggestEntities}
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
          <button type="button" className="btn-ds btn-ds-secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="btn-ds btn-ds-primary" onClick={submit} disabled={busy}>
            Ingest Triplet
          </button>
        </div>
      </div>
    </div>
  );
}
