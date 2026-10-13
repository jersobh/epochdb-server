export type NodeType =
  | 'person'
  | 'organization'
  | 'place'
  | 'system'
  | 'artifact'
  | 'event'
  | 'concept'
  | 'skill'
  | 'rule'
  | 'step';

export const TYPE_STYLES: Record<
  NodeType,
  { label: string; light: string; dark: string; shape: string }
> = {
  person: { label: 'Person', light: '#2563EB', dark: '#60A5FA', shape: 'dot' },
  organization: { label: 'Organization', light: '#7C3AED', dark: '#A78BFA', shape: 'square' },
  place: { label: 'Place', light: '#059669', dark: '#34D399', shape: 'triangle' },
  system: { label: 'System / Tech', light: '#0891B2', dark: '#22D3EE', shape: 'hexagon' },
  artifact: { label: 'Invention / Work', light: '#D97706', dark: '#FBBF24', shape: 'diamond' },
  event: { label: 'Event / Award', light: '#E11D48', dark: '#FB7185', shape: 'star' },
  concept: { label: 'Concept', light: '#64748B', dark: '#94A3B8', shape: 'dot' },
  skill: { label: 'Skill', light: '#DB2777', dark: '#F472B6', shape: 'box' },
  rule: { label: 'Rule', light: '#CA8A04', dark: '#FACC15', shape: 'triangleDown' },
  step: { label: 'Step', light: '#78716C', dark: '#A8A29E', shape: 'ellipse' },
};

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
  const st = TYPE_STYLES[type as NodeType] || TYPE_STYLES.concept;
  const useDark = dark ?? isDarkMode();
  return useDark ? st.dark : st.light;
}

export function classifyNodes(
  names: string[],
  edges: RawEdge[],
  skillNames: Set<string>
): Record<string, NodeType> {
  const votes: Record<string, Record<string, number>> = {};
  const vote = (n: string, t: NodeType | null) => {
    if (!t) return;
    if (!votes[n]) votes[n] = {};
    votes[n][t] = (votes[n][t] || 0) + 1;
  };

  (edges || []).forEach((e) => {
    const role = PRED_ROLES[normPred(e.predicate || e.label)];
    if (role) {
      vote(edgeSource(e), role[0]);
      vote(edgeTarget(e), role[1]);
    }
  });

  const out: Record<string, NodeType> = {};
  names.forEach((n) => {
    const l = n.toLowerCase();
    if (/^skill:/.test(l) || skillNames.has(l)) out[n] = 'skill';
    else if (/^rule:/.test(l)) out[n] = 'rule';
    else if (/^step:/.test(l)) out[n] = 'step';
    else if (votes[n]) {
      out[n] = Object.entries(votes[n]).sort((a, b) => b[1] - a[1])[0][0] as NodeType;
    } else out[n] = 'concept';
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
