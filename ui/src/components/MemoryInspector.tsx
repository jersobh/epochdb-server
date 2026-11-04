import { useMemo } from 'react';
import type { SkillItem } from '../types';
import type { NodeType, RawEdge } from '../lib/graph';
import { TYPE_STYLES, edgeSource, edgeTarget } from '../lib/graph';

interface MemoryInspectorProps {
  open: boolean;
  nodeId: string | null;
  nodeType: NodeType | string;
  edges: RawEdge[];
  skills: SkillItem[];
  onClose: () => void;
}

function skillName(s: SkillItem): string {
  return String(s.skill_name || s.metadata?.skill_name || s.id || '');
}

export function MemoryInspector({
  open,
  nodeId,
  nodeType,
  edges,
  skills,
  onClose,
}: MemoryInspectorProps) {
  const outgoing: string[] = [];
  const incoming: string[] = [];
  if (nodeId) {
    edges.forEach((e) => {
      const s = edgeSource(e);
      const t = edgeTarget(e);
      const p = e.predicate || e.label || 'relates';
      if (s === nodeId) outgoing.push(`→ [${p}] → ${t}`);
      if (t === nodeId) incoming.push(`${s} → [${p}] →`);
    });
  }

  const relatedSkills = useMemo(() => {
    if (!nodeId) return [];
    const nLow = nodeId.toLowerCase().trim();
    return skills.filter((s) => {
      const name = skillName(s).toLowerCase().trim();
      if (name === nLow) return true;
      const rawSteps = (s.steps || s.metadata?.steps || []) as Array<{ step?: number; action?: string; description?: string } | string>;
      const stepMatch = rawSteps.some((st) => {
        const text = typeof st === 'string' ? st : (st.action || st.description || '');
        return text.toLowerCase().trim() === nLow;
      });
      if (stepMatch) return true;
      return edges.some((e) => {
        const src = edgeSource(e).toLowerCase().trim();
        const tgt = edgeTarget(e).toLowerCase().trim();
        return (src === nLow && tgt === name) || (tgt === nLow && src === name);
      });
    });
  }, [nodeId, skills, edges]);

  const typeLabel = (TYPE_STYLES[nodeType as NodeType] || TYPE_STYLES.concept).label;

  return (
    <div className={`drawer-overlay${open ? ' active' : ''}`} onClick={onClose}>
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <h2 style={{ fontSize: 15, fontWeight: 700, fontFamily: 'var(--font-sans)' }}>
              {nodeId || 'Memory Inspector'}
            </h2>
            <p style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Explore connected triples, procedural skills, and atomic memories
            </p>
          </div>
          <button type="button" className="close-btn" onClick={onClose}>
            &times;
          </button>
        </div>

        <div
          style={{
            padding: '16px 20px',
            overflowY: 'auto',
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
          }}
        >
          <div className="metric-card">
            <div className="metric-title">
              <span style={{ fontSize: 14, fontWeight: 700 }}>{nodeId || 'Entity'}</span>
              <span className="scope-badge" style={{ fontSize: 11, cursor: 'default' }}>
                {typeLabel}
              </span>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
              {nodeId
                ? `Inspecting relationships for ${nodeId}.`
                : 'Select a node in the graph or choose a memory to inspect relationships.'}
            </div>
          </div>

          <div>
            <h3 style={{ fontSize: 12, fontWeight: 700, marginBottom: 8, color: 'var(--text-secondary)' }}>
              Direct Relationships
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {outgoing.length === 0 && incoming.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No direct triples recorded.</div>
              ) : (
                [...outgoing, ...incoming].map((r) => (
                  <div
                    key={r}
                    className="scope-badge"
                    style={{ width: '100%', fontFamily: 'var(--font-mono)', cursor: 'default' }}
                  >
                    {r}
                  </div>
                ))
              )}
            </div>
          </div>

          <div>
            <h3 style={{ fontSize: 12, fontWeight: 700, marginBottom: 8, color: 'var(--text-secondary)' }}>
              {nodeType === 'skill' ? 'Procedural Skill Definition' : 'Linked Procedural Skills'}
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {relatedSkills.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  No procedural skills linked to this entity.
                </div>
              ) : (
                relatedSkills.map((s, i) => {
                  const name = skillName(s);
                  const desc = String(s.description || s.metadata?.description || '');
                  const rawSteps = (s.steps || s.metadata?.steps || []) as Array<{ step?: number; action?: string; description?: string } | string>;
                  return (
                    <div key={`${name}-${i}`} className="metric-card">
                      <div className="metric-title">
                        <span style={{ fontWeight: 600 }}>{name}</span>
                        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}>
                          {rawSteps.length} steps
                        </span>
                      </div>
                      {desc && (
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: rawSteps.length > 0 ? 8 : 0 }}>
                          {desc}
                        </div>
                      )}
                      {rawSteps.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
                          {rawSteps.map((st, sIdx) => {
                            const act = typeof st === 'string' ? st : (st.action || st.description || `Step ${sIdx + 1}`);
                            return (
                              <div
                                key={sIdx}
                                style={{
                                  fontSize: 11,
                                  fontFamily: 'var(--font-mono)',
                                  color: 'var(--text-secondary)',
                                  background: 'var(--surface-hover)',
                                  padding: '4px 8px',
                                  borderRadius: 4,
                                }}
                              >
                                {sIdx + 1}. {act}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
