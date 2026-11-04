import {
  ENTITY_TYPE_LABELS,
  ENTITY_TYPES,
  PRED_ROLES,
  TYPE_DEFINING_PREDICATES,
  inferTypeFromName,
  normalizeType,
  type NodeType,
} from './ontology';

export type { NodeType } from './ontology';
export { ENTITY_TYPES, ENTITY_TYPE_LABELS, PRED_ROLES, normalizeType } from './ontology';

// Base styles for closed ontology types (+ entity fallback)
export const BASE_TYPE_STYLES: Record<string, { label: string; light: string; dark: string; shape: string }> = {
  person: { label: 'Person', light: '#2563EB', dark: '#60A5FA', shape: 'dot' },
  organization: { label: 'Organization', light: '#7C3AED', dark: '#A78BFA', shape: 'square' },
  role: { label: 'Role', light: '#0284C7', dark: '#38BDF8', shape: 'dot' },
  agent: { label: 'Agent', light: '#0D9488', dark: '#2DD4BF', shape: 'hexagon' },
  product: { label: 'Product', light: '#C026D3', dark: '#E879F9', shape: 'box' },
  service: { label: 'Service', light: '#9333EA', dark: '#C084FC', shape: 'box' },
  feature: { label: 'Feature', light: '#A21CAF', dark: '#F0ABFC', shape: 'diamond' },
  artifact: { label: 'Artifact', light: '#D97706', dark: '#FBBF24', shape: 'diamond' },
  system: { label: 'System', light: '#0891B2', dark: '#22D3EE', shape: 'hexagon' },
  dataset: { label: 'Dataset', light: '#0E7490', dark: '#67E8F9', shape: 'box' },
  place: { label: 'Place', light: '#059669', dark: '#34D399', shape: 'triangle' },
  event: { label: 'Event', light: '#E11D48', dark: '#FB7185', shape: 'star' },
  market: { label: 'Market', light: '#CA8A04', dark: '#FACC15', shape: 'triangle' },
  channel: { label: 'Channel', light: '#4F46E5', dark: '#818CF8', shape: 'ellipse' },
  campaign: { label: 'Campaign', light: '#DB2777', dark: '#F472B6', shape: 'star' },
  concept: { label: 'Concept', light: '#64748B', dark: '#94A3B8', shape: 'dot' },
  skill: { label: 'Skill', light: '#DB2777', dark: '#F472B6', shape: 'box' },
  process: { label: 'Process', light: '#78716C', dark: '#A8A29E', shape: 'ellipse' },
  metric: { label: 'Metric', light: '#EA580C', dark: '#FB923C', shape: 'triangleDown' },
  evidence: { label: 'Evidence', light: '#16A34A', dark: '#4ADE80', shape: 'dot' },
  hypothesis: { label: 'Hypothesis', light: '#B45309', dark: '#FBBF24', shape: 'diamond' },
  session: { label: 'Session', light: '#475569', dark: '#94A3B8', shape: 'ellipse' },
  entity: { label: 'Entity', light: '#475569', dark: '#94A3B8', shape: 'dot' },
};

export function getTypeStyle(type: string | undefined): { label: string; light: string; dark: string; shape: string } {
  const t = normalizeType(type, 'entity');
  if (BASE_TYPE_STYLES[t]) {
    return BASE_TYPE_STYLES[t];
  }
  const label = ENTITY_TYPE_LABELS[t] || (type || 'Entity');
  return {
    label,
    light: '#475569',
    dark: '#94A3B8',
    shape: 'dot',
  };
}

export const TYPE_STYLES: Record<string, { label: string; light: string; dark: string; shape: string }> = new Proxy(
  BASE_TYPE_STYLES,
  {
    get(target, prop: string) {
      if (typeof prop === 'string') {
        return getTypeStyle(prop);
      }
      return target[prop as keyof typeof target];
    },
  }
);

export const DEFAULT_PREDICATES = [
  'is_a',
  'holds_role',
  'works_at',
  'member_of',
  'located_in',
  'authored',
  'mentioned_in',
  'participated_in',
  'published_on',
  'about',
  'observed',
  'hypothesized',
  'works_on',
  'uses',
  'part_of',
];

export interface RawEdge {
  source?: string | { id: string };
  target?: string | { id: string };
  from?: string;
  to?: string;
  predicate?: string;
  label?: string;
}

export function edgeSource(e: RawEdge): string {
  if (typeof e.source === 'object' && e.source) return String(e.source.id);
  return String(e.source || e.from || '');
}

export function edgeTarget(e: RawEdge): string {
  if (typeof e.target === 'object' && e.target) return String(e.target.id);
  return String(e.target || e.to || '');
}

export function normPred(p: string | undefined): string {
  return String(p || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_');
}

export function isDarkMode(): boolean {
  return document.body.classList.contains('dark-mode');
}

export function typeColor(type: string, dark?: boolean): string {
  const st = getTypeStyle(type);
  const useDark = dark ?? isDarkMode();
  return useDark ? st.dark : st.light;
}

/**
 * Classify nodes using registered ∪ discovered types when provided.
 * Auto-discovery from is_a triples stays open; job titles / colon junk are not types.
 */
export function classifyNodes(
  names: string[],
  edges: RawEdge[],
  skillNames: Set<string>,
  knownTypes?: Set<string>,
  serverTypes?: Record<string, string>
): Record<string, NodeType> {
  const explicitTypes: Record<string, string> = {};
  const relationalVotes: Record<string, Record<string, number>> = {};
  const allowed = knownTypes && knownTypes.size > 0 ? knownTypes : undefined;

  const accept = (raw: string | null | undefined): string | null => {
    if (!raw) return null;
    // Prefer alias/seed normalization, then compact open ids (auto-discover).
    const mapped = normalizeType(raw, '');
    if (mapped && mapped !== 'entity') return mapped;
    const slug = String(raw)
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, '_');
    if (slug.length <= 64 && /^[a-z][a-z0-9_]*$/.test(slug)) {
      if (!allowed || allowed.has(slug)) return slug;
      // New discovery not yet registered — still accept compact ids
      return slug;
    }
    return null;
  };

  const vote = (n: string, t: string | null) => {
    if (!t || !n) return;
    const canon = accept(t);
    if (!canon) return;
    if (!relationalVotes[n]) relationalVotes[n] = {};
    relationalVotes[n][canon] = (relationalVotes[n][canon] || 0) + 1;
  };

  if (serverTypes) {
    Object.entries(serverTypes).forEach(([nid, stype]) => {
      const canon = accept(stype);
      if (canon && canon !== 'entity') {
        explicitTypes[nid] = canon;
      }
    });
  }

  (edges || []).forEach((e) => {
    const s = edgeSource(e);
    const t = edgeTarget(e);
    const pred = normPred(e.predicate || e.label);

    if (TYPE_DEFINING_PREDICATES.has(pred) && t) {
      const mapped = accept(t);
      if (mapped) explicitTypes[s] = mapped;
    }

    if (pred === 'is_skill') {
      explicitTypes[s] = 'skill';
      explicitTypes[t] = 'skill';
    } else if (pred.startsWith('has_step')) {
      explicitTypes[s] = 'skill';
      vote(t, 'process');
    }

    const role = PRED_ROLES[pred];
    if (role) {
      if (role[0]) vote(s, role[0]);
      if (role[1]) vote(t, role[1]);
    } else if (pred && pred !== 'related' && pred !== 'co_occurs_with' && pred !== 'mentions') {
      if (/^(located|headquartered)_in$/.test(pred) || pred === 'born_in' || pred === 'lives_in') {
        vote(t, 'place');
      } else if (/works|employed|leads|manages|founded/.test(pred)) {
        vote(s, 'person');
        vote(t, 'organization');
      } else if (/uses|runs|requires|depends|integrates/.test(pred)) {
        vote(t, 'system');
      } else if (/invented|created|designed|patented|formulated|discovered/.test(pred)) {
        vote(s, 'person');
      } else if (/powers|enables|supports/.test(pred)) {
        vote(t, 'system');
      }
    }
  });

  const out: Record<string, NodeType> = {};
  names.forEach((n) => {
    const l = n.toLowerCase();
    if ((/^skill:/.test(l) || skillNames.has(l)) && (!allowed || allowed.has('skill'))) {
      out[n] = 'skill';
      return;
    }

    if (explicitTypes[n]) {
      out[n] = explicitTypes[n];
      return;
    }

    const fromName = inferTypeFromName(n);
    if (fromName && (!allowed || allowed.has(fromName))) {
      out[n] = fromName;
      return;
    }

    if (relationalVotes[n]) {
      out[n] = Object.entries(relationalVotes[n]).sort((a, b) => b[1] - a[1])[0][0];
      return;
    }

    out[n] = 'entity';
  });

  return out;
}
export function computeDegrees(names: string[], edges: RawEdge[]): Record<string, number> {
  const degs: Record<string, number> = {};
  names.forEach((n) => {
    degs[n] = 0;
  });
  edges.forEach((e) => {
    const s = edgeSource(e);
    const t = edgeTarget(e);
    if (degs[s] !== undefined) degs[s]++;
    if (degs[t] !== undefined) degs[t]++;
  });
  return degs;
}

/** Legend order: ontology order, then entity last if present. */
export function sortTypeLegend(types: string[]): string[] {
  const order = new Map<string, number>(ENTITY_TYPES.map((t, i) => [t, i]));
  return [...types].sort((a, b) => {
    const ia = order.has(a) ? (order.get(a) as number) : 1000;
    const ib = order.has(b) ? (order.get(b) as number) : 1000;
    if (a === 'entity') return 1;
    if (b === 'entity') return -1;
    if (ia !== ib) return ia - ib;
    return a.localeCompare(b);
  });
}
