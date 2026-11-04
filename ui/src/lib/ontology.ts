/** Closed entity-type ontology — keep in sync with src/ontology.py */

export const ENTITY_TYPES = [
  'person',
  'organization',
  'role',
  'agent',
  'product',
  'service',
  'feature',
  'artifact',
  'system',
  'dataset',
  'place',
  'event',
  'market',
  'channel',
  'campaign',
  'concept',
  'skill',
  'process',
  'metric',
  'evidence',
  'hypothesis',
  'session',
] as const;

export type CanonicalType = (typeof ENTITY_TYPES)[number];
export type NodeType = CanonicalType | 'entity' | string;

export const ENTITY_TYPE_SET = new Set<string>(ENTITY_TYPES);

export const ENTITY_TYPE_LABELS: Record<string, string> = {
  person: 'Person',
  organization: 'Organization',
  role: 'Role',
  agent: 'Agent',
  product: 'Product',
  service: 'Service',
  feature: 'Feature',
  artifact: 'Artifact',
  system: 'System',
  dataset: 'Dataset',
  place: 'Place',
  event: 'Event',
  market: 'Market',
  channel: 'Channel',
  campaign: 'Campaign',
  concept: 'Concept',
  skill: 'Skill',
  process: 'Process',
  metric: 'Metric',
  evidence: 'Evidence',
  hypothesis: 'Hypothesis',
  session: 'Session',
  entity: 'Entity',
};

const TYPE_ALIASES: Record<string, string> = {
  people: 'person',
  human: 'person',
  contact: 'person',
  user: 'person',
  company: 'organization',
  org: 'organization',
  employer: 'organization',
  team: 'organization',
  community: 'organization',
  job_title: 'role',
  title: 'role',
  position: 'role',
  bot: 'agent',
  ai_agent: 'agent',
  sku: 'product',
  offering: 'product',
  tool: 'system',
  platform: 'system',
  software: 'system',
  infra: 'system',
  infrastructure: 'system',
  model: 'system',
  api: 'system',
  doc: 'artifact',
  document: 'artifact',
  post: 'artifact',
  article: 'artifact',
  paper: 'artifact',
  repo: 'artifact',
  repository: 'artifact',
  media: 'artifact',
  creative_work: 'artifact',
  location: 'place',
  city: 'place',
  country: 'place',
  venue: 'place',
  region: 'place',
  meetup: 'event',
  conference: 'event',
  interview: 'event',
  vertical: 'market',
  segment: 'market',
  industry: 'market',
  linkedin: 'channel',
  twitter: 'channel',
  x: 'channel',
  email: 'channel',
  call: 'channel',
  topic: 'concept',
  domain: 'concept',
  idea: 'concept',
  competency: 'skill',
  workflow: 'process',
  playbook: 'process',
  kpi: 'metric',
  fact: 'evidence',
  observation: 'evidence',
  inference: 'hypothesis',
  claim: 'hypothesis',
  observe_session: 'session',
  experience: 'role',
  headline: 'artifact',
  about: 'artifact',
  visible_post: 'artifact',
  visible_quote: 'artifact',
  post_themes: 'concept',
  post_theme: 'concept',
  followers: 'metric',
  follower_count: 'metric',
  education: 'artifact',
  current_employer: 'organization',
  lead: 'role',
  rule: 'process',
  step: 'process',
  stakeholder: 'person',
  champion: 'person',
  economic_buyer: 'person',
  technical_gatekeeper: 'person',
};

export const TYPE_DEFINING_PREDICATES = new Set([
  'is_a',
  'instance_of',
  'rdf:type',
  'type',
  'has_type',
  'entity_type',
  'classified_as',
  'subclass_of',
  'kind_of',
]);

export const PRED_ROLES: Record<string, [NodeType | null, NodeType | null]> = {
  born_in: ['person', 'place'],
  lives_in: ['person', 'place'],
  died_in: ['person', 'place'],
  works_at: ['person', 'organization'],
  worked_at: ['person', 'organization'],
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
  mentioned_in: ['person', 'artifact'],
  participated_in: ['person', 'event'],
  published_on: ['artifact', 'channel'],
  about: [null, 'concept'],
  holds_role: ['person', 'role'],
  has_label: ['person', null],
  observed: ['person', 'evidence'],
  hypothesized: ['person', 'hypothesis'],
  works_on: ['person', 'product'],
  builds: ['person', 'product'],
  sells: ['organization', 'product'],
  offers: ['organization', 'service'],
  has_feature: ['product', 'feature'],
  uses: ['person', 'system'],
  runs_on: ['system', 'system'],
  located_in: [null, 'place'],
  headquartered_in: ['organization', 'place'],
  competes_with: ['organization', 'organization'],
  part_of: [null, 'organization'],
  invented: ['person', 'product'],
  commercialized: ['organization', 'system'],
  powers: ['product', 'system'],
  developed: ['person', 'concept'],
  created: ['person', 'artifact'],
  designed: ['person', 'artifact'],
  discovered: ['person', 'concept'],
  formulated: ['person', 'concept'],
  patented: ['person', 'product'],
  produces: ['organization', 'product'],
  manufactures: ['organization', 'product'],
  enables: ['concept', 'system'],
  influences: ['concept', 'system'],
  implements: ['system', 'concept'],
  distributes_data_to: ['system', 'system'],
  routes_queries_to: ['system', 'system'],
  compresses_epochs_into: ['system', 'artifact'],
  supports_queries_via: ['artifact', 'system'],
  analyzes: ['system', 'system'],
  generates: ['system', 'dataset'],
  combines: ['system', 'dataset'],
  is_skill: ['skill', 'skill'],
  has_step: ['skill', 'process'],
  wrote_first_algorithm_for: ['person', 'artifact'],
  named_after: [null, 'person'],
  led_by: ['organization', 'person'],
  ceo_of: ['person', 'organization'],
  co_founded: ['person', 'organization'],
  pioneered: ['person', 'concept'],
  fellow_at: ['person', 'organization'],
  rival_of: ['person', 'person'],
  mentored: ['person', 'person'],
  cracked: ['person', 'system'],
  built: ['organization', 'system'],
  manages: ['system', 'system'],
  powered_by: ['product', 'product'],
  launches: ['product', 'system'],
  provides: ['system', 'service'],
  connects_to: ['service', 'system'],
  supports: ['organization', 'system'],
};

function slug(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/-/g, '_')
    .replace(/\s+/g, '_')
    .replace(/\//g, '_');
}

export function normalizeType(raw: string | undefined | null, fallback: string = 'entity'): string {
  if (raw == null) return fallback;
  const s = slug(raw);
  if (!s) return fallback;
  if (ENTITY_TYPE_SET.has(s)) return s;
  if (TYPE_ALIASES[s]) return TYPE_ALIASES[s];
  if (String(raw).includes(':')) {
    const prefix = slug(String(raw).split(':', 1)[0]);
    if (ENTITY_TYPE_SET.has(prefix)) return prefix;
    if (TYPE_ALIASES[prefix]) return TYPE_ALIASES[prefix];
  }
  return fallback;
}

export function inferTypeFromName(name: string): string | null {
  const n = String(name || '').trim();
  if (!n) return null;
  const low = n.toLowerCase();
  if (low.startsWith('person_') || low === 'person') return 'person';
  if (low.startsWith('name_') || low.startsWith('email_') || low.startsWith('phone_')) return 'person';
  if (low.startsWith('org_') || low.startsWith('company_')) return 'organization';
  if (/\b(university|college|institute|foundation|electric|corp|inc|ltd|gmbh)\b/i.test(low)) return 'organization';
  if (low.startsWith('product_')) return 'product';
  if (low.startsWith('session_')) return 'session';
  if (low.startsWith('skill:') || low.startsWith('skill ')) return 'skill';
  if (low === 'evidence' || low === 'hypothesis') return low;
  if (low.startsWith('linkedin') || ` ${low} `.includes(' linkedin ')) return 'artifact';
  if (n.includes(':')) {
    const mapped = normalizeType(n.split(':', 1)[0].trim(), '');
    return mapped || null;
  }
  return null;
}

export function isCanonicalType(t: string): boolean {
  return ENTITY_TYPE_SET.has(normalizeType(t, ''));
}
