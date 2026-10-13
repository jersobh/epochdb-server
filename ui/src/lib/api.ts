import type {
  ClusterStats,
  ExportBundle,
  GraphPayload,
  ProfileItem,
  ServerConfig,
  SkillItem,
} from '../types';

export type AuthRequiredHandler = () => void;

let onAuthRequired: AuthRequiredHandler | null = null;

export function setAuthRequiredHandler(handler: AuthRequiredHandler | null) {
  onAuthRequired = handler;
}

export function getServerConfig(): ServerConfig {
  return window.__EPOCHDB_SERVER_CONFIG__ || {};
}

export function getToken(): string {
  const srv = getServerConfig();
  return (
    localStorage.getItem('epochdb_token') ||
    srv.defaultApiKey ||
    srv.defaultInternalToken ||
    ''
  );
}

export function getTenant(): string {
  return (localStorage.getItem('epochdb_tenant') || 'global').trim();
}

export function getNamespace(): string {
  return (localStorage.getItem('epochdb_namespace') || 'default').trim();
}

export function saveAuth(token: string, tenant: string, namespace: string) {
  localStorage.setItem('epochdb_token', token);
  localStorage.setItem('epochdb_tenant', tenant || 'global');
  localStorage.setItem('epochdb_namespace', namespace || 'default');
}

export function ensureDefaultToken() {
  const srv = getServerConfig();
  if (!localStorage.getItem('epochdb_token') && (srv.defaultApiKey || srv.defaultInternalToken)) {
    localStorage.setItem('epochdb_token', srv.defaultApiKey || srv.defaultInternalToken || '');
  }
}

function authHeaders(): HeadersInit {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['X-API-Key'] = token;
    headers['X-Internal-Token'] = token;
  }
  const tenant = getTenant();
  const namespace = getNamespace();
  if (tenant) headers['X-Tenant-ID'] = tenant;
  if (namespace) headers['X-Namespace'] = namespace;
  return headers;
}

export async function apiRequest<T = unknown>(
  endpoint: string,
  method: string = 'GET',
  body: unknown = null
): Promise<T> {
  const config: RequestInit = {
    method,
    headers: authHeaders(),
  };
  if (body != null) config.body = JSON.stringify(body);

  const res = await fetch(endpoint, config);
  if (res.status === 401) {
    onAuthRequired?.();
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({} as { detail?: string }));
    throw new Error(err.detail || 'API request failed');
  }
  return res.json() as Promise<T>;
}

export async function fetchHealthz(): Promise<{ mode?: string; version?: string }> {
  return fetch('/healthz').then((r) => r.json()).catch(() => ({ mode: 'coordinator' }));
}

export async function fetchEntityGraph(entityId?: string, depth = 2): Promise<GraphPayload> {
  const params = new URLSearchParams();
  if (entityId) params.set('entity_id', entityId);
  params.set('depth', String(depth));
  return apiRequest(`/entity_graph?${params.toString()}`);
}

export async function fetchStats(): Promise<ClusterStats> {
  return apiRequest('/stats');
}

export async function fetchSkills(): Promise<SkillItem[]> {
  const res = await apiRequest<{ skills?: SkillItem[] }>('/skills');
  return res.skills || [];
}

export async function addTriplet(subject: string, predicate: string, object: string) {
  return apiRequest('/remember', 'POST', {
    text: `${subject} ${predicate.replace(/_/g, ' ')} ${object}.`,
    subject,
    predicate,
    object,
    triples: [[subject, predicate, object]],
  });
}

export async function rememberTriple(subject: string, predicate: string, object: string) {
  return apiRequest('/remember', 'POST', {
    text: `${subject} ${predicate.replace(/_/g, ' ')} ${object}.`,
    triples: [[subject, predicate, object]],
  });
}

export async function rememberSkill(payload: {
  skill_name: string;
  description: string;
  steps: Array<{ step: number; action: string }>;
}) {
  return apiRequest('/skills/remember', 'POST', payload);
}

export async function exportData(dataType = 'all'): Promise<ExportBundle> {
  return apiRequest(`/export?data_type=${encodeURIComponent(dataType)}`);
}

export async function importData(data: unknown, dataType = 'all') {
  return apiRequest('/import', 'POST', { data_type: dataType, data });
}

export async function fetchProfiles(): Promise<ProfileItem[]> {
  const b = await exportData('profiles');
  return b.data?.profiles || [];
}

export async function recallQuery(query: string, k = 5) {
  return apiRequest<{ memories?: Array<{ payload?: string; text?: string; score?: number; similarity?: number }> }>(
    '/query',
    'POST',
    { query, k }
  );
}

export function connectSSE(handlers: {
  onUpdate?: () => void;
  onStats?: (data: ClusterStats) => void;
}): EventSource {
  const es = new EventSource('/stream');
  es.addEventListener('update', () => {
    handlers.onUpdate?.();
  });
  es.addEventListener('stats', (e) => {
    try {
      const data = JSON.parse((e as MessageEvent).data);
      handlers.onStats?.(data);
    } catch {
      /* ignore */
    }
  });
  return es;
}
