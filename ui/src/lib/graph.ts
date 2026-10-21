export type NodeType = string;

// Base styles for standard types, while fully supporting any dynamically discovered type
export const BASE_TYPE_STYLES: Record<string, { label: string; light: string; dark: string; shape: string }> = {
  person: { label: 'Person', light: '#2563EB', dark: '#60A5FA', shape: 'dot' },
  organization: { label: 'Organization', light: '#7C3AED', dark: '#A78BFA', shape: 'square' },
  company: { label: 'Company', light: '#4F46E5', dark: '#818CF8', shape: 'square' },
  place: { label: 'Place', light: '#059669', dark: '#34D399', shape: 'triangle' },
  system: { label: 'System / Tech', light: '#0891B2', dark: '#22D3EE', shape: 'hexagon' },
  artifact: { label: 'Invention / Work', light: '#D97706', dark: '#FBBF24', shape: 'diamond' },
  event: { label: 'Event / Award', light: '#E11D48', dark: '#FB7185', shape: 'star' },
  concept: { label: 'Concept', light: '#64748B', dark: '#94A3B8', shape: 'dot' },
  entity: { label: 'Entity', light: '#475569', dark: '#94A3B8', shape: 'dot' },
  skill: { label: 'Skill', light: '#DB2777', dark: '#F472B6', shape: 'box' },
  rule: { label: 'Rule', light: '#CA8A04', dark: '#FACC15', shape: 'triangleDown' },
  step: { label: 'Step', light: '#78716C', dark: '#A8A29E', shape: 'ellipse' },
  lead: { label: 'Lead', light: '#0284C7', dark: '#38BDF8', shape: 'dot' },
  stakeholder: { label: 'Stakeholder', light: '#10B981', dark: '#34D399', shape: 'dot' },
  champion: { label: 'Champion', light: '#059669', dark: '#10B981', shape: 'star' },
  economic_buyer: { label: 'Economic Buyer', light: '#6366F1', dark: '#818CF8', shape: 'square' },
  technical_gatekeeper: { label: 'Technical Gatekeeper', light: '#D97706', dark: '#FBBF24', shape: 'diamond' },
};

// Generates dynamic type style for ANY discovered type on-the-fly
export function getTypeStyle(type: string | undefined): { label: string; light: string; dark: string; shape: string } {
  const t = (type || 'entity').trim().toLowerCase().replace(/[\s-]+/g, '_');
  if (BASE_TYPE_STYLES[t]) {
    return BASE_TYPE_STYLES[t];
  }

  // Format human-readable title
  const label = (type || 'Entity')
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());

  // Deterministic 32-bit hash for consistent palette across sessions
  let hash = 0;
  for (let i = 0; i < t.length; i++) {
    hash = (hash << 5) - hash + t.charCodeAt(i);
    hash |= 0;
  }
  const hue = Math.abs(hash) % 360;
  const shapes = ['dot', 'square', 'triangle', 'hexagon', 'diamond', 'star', 'box', 'ellipse'];
  const shape = shapes[Math.abs(hash >> 3) % shapes.length];

  return {
    label,
    light: `hsl(${hue}, 70%, 42%)`,
    dark: `hsl(${hue}, 80%, 65%)`,
    shape,
  };
}

// Transparent Proxy so TYPE_STYLES[anyType] always resolves dynamically
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

export const PRED_ROLES: Record<string, [NodeType | null, NodeType | null]> = {
  born_in: ['person', 'place'],
  lives_in: ['person', 'place'],
  died_in: ['person', 'place'],
  works_at: ['person', 'organization'],
  employed_by: ['person', 'organization'],
  studied_at: ['person', 'organization'],
  member_of: ['person', 'organization'],
  leads: ['person', 'organization'],
  founded: ['person', 'organization'],
  founded_by: ['organization', 'person'],
  collaborated_with: ['person', 'person'],
  won: ['person', 'event'],
  wrote: ['person', 'artifact'],
  authored: ['person', 'artifact'],
  discovered: ['person', 'concept'],
  pioneered: ['person', 'concept'],
  invented: [null, 'artifact'],
  created: [null, 'artifact'],
  designed: [null, 'artifact'],
  developed: [null, 'concept'],
  created_by: ['artifact', null],
  designed_by: ['artifact', null],
  commercialized: ['organization', 'artifact'],
  competes_with: ['organization', 'organization'],
  headquartered_in: ['organization', 'place'],
  located_in: [null, 'place'],
  capital_of: ['place', 'place'],
  depends_on: ['system', 'system'],
  relies_on: ['system', 'system'],
  runs_on: ['system', 'system'],
  deployed_on: ['system', 'system'],
  orchestrates: ['system', 'system'],
  calls: ['system', 'system'],
  monitors: ['system', 'system'],
  visualizes: ['system', 'system'],
  hosts: ['system', 'system'],
  powers: [null, 'system'],
  implements: [null, 'concept'],
  enables: [null, 'concept'],
  treats: ['artifact', 'concept'],
  prevents: ['artifact', 'concept'],
  operates_in: ['organization', 'market'],
  has_stakeholder: ['organization', 'person'],
  holds_role: ['person', 'role'],
};

export const DEFAULT_PREDICATES = [
  'relies_on',
  'depends_on',
  'implements',
  'part_of',
  'located_in',
  'won',
  'created_by',
  'invented',
  'works_at',
  'born_in',
  'operates_in',
  'holds_role',
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

// Predicates that explicitly designate entity type/classification in the graph
const TYPE_DEFINING_PREDICATES = new Set([
  'is_a',
  'type',
  'entity_type',
  'has_type',
  'category',
  'role',
  'holds_role',
  'classified_as',
  'subclass_of',
  'instance_of',
  'kind_of',
]);

export function classifyNodes(
  names: string[],
  edges: RawEdge[],
  skillNames: Set<string>
): Record<string, NodeType> {
  const explicitTypes: Record<string, string> = {};
  const relationalVotes: Record<string, Record<string, number>> = {};

  const vote = (n: string, t: string | null) => {
    if (!t) return;
    if (!relationalVotes[n]) relationalVotes[n] = {};
    relationalVotes[n][t] = (relationalVotes[n][t] || 0) + 1;
  };

  (edges || []).forEach((e) => {
    const s = edgeSource(e);
    const t = edgeTarget(e);
    const pred = normPred(e.predicate || e.label);

    // 1. Explicit ontological discovery from triples: (Entity, is_a | type | holds_role, TargetType)
    if (TYPE_DEFINING_PREDICATES.has(pred)) {
      if (t) {
        explicitTypes[s] = t;
      }
    }

    // 2. Relational discovery from known predicate roles
    const role = PRED_ROLES[pred];
    if (role) {
      if (role[0]) vote(s, role[0]);
      if (role[1]) vote(t, role[1]);
    } else if (pred) {
      // Dynamic relational inference
      if (/in|at|located|headquartered/.test(pred)) {
        vote(t, 'place');
      } else if (/works|employed|leads|manages|founded/.test(pred)) {
        vote(s, 'person');
        vote(t, 'organization');
      } else if (/uses|runs|requires|depends|integrates/.test(pred)) {
        vote(t, 'system');
      }
    }
  });

  const out: Record<string, NodeType> = {};
  names.forEach((n) => {
    const l = n.toLowerCase();

    // 1. Explicit type discovered in triples
    if (explicitTypes[n]) {
      out[n] = explicitTypes[n];
      return;
    }

    // 2. Prefixes (e.g., skill:..., rule:..., lead:..., company:...)
    if (/^skill:/.test(l) || skillNames.has(l)) {
      out[n] = 'skill';
    } else if (/^rule:/.test(l)) {
      out[n] = 'rule';
    } else if (/^step:/.test(l)) {
      out[n] = 'step';
    } else if (n.includes(':')) {
      const prefix = n.split(':')[0].trim().toLowerCase();
      if (prefix && prefix.length <= 16) {
        out[n] = prefix;
        return;
      }
    }

    // 3. Relational discovery votes
    if (!out[n] && relationalVotes[n]) {
      out[n] = Object.entries(relationalVotes[n]).sort((a, b) => b[1] - a[1])[0][0];
    }

    // 4. Default discovered general entity type
    if (!out[n]) {
      out[n] = 'entity';
    }
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
