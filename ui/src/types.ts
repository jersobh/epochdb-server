import type { NodeType, RawEdge } from './lib/graph';

export type { NodeType, RawEdge };

export interface ServerConfig {
  mode?: string;
  version?: string;
  defaultApiKey?: string;
  defaultInternalToken?: string;
}

declare global {
  interface Window {
    __EPOCHDB_SERVER_CONFIG__?: ServerConfig;
  }
}

export interface GraphPayload {
  nodes: string[];
  edges: RawEdge[];
}

export interface SkillItem {
  id?: string;
  skill_name?: string;
  description?: string;
  steps?: Array<{ step?: number; action: string; [key: string]: unknown } | string>;
  tool_schema?: Record<string, unknown>;
  decision_rules?: string[];
  triples?: Array<[string, string, string]>;
  created_at?: number;
  metadata?: Record<string, unknown>;
  payload?: string;
}

export interface ProfileItem {
  id?: string;
  user_id?: string;
  fact_text?: string;
  payload?: string;
  text?: string;
  created_at?: number;
  metadata?: Record<string, unknown>;
}

export interface ShardMetric {
  status: 'healthy' | 'unhealthy' | 'down' | string;
  cpu?: number;
  ram?: { total: number; available: number; used: number; percent: number };
  disk?: { total: number; available: number; used: number; percent: number };
  memory_count?: number;
  l1_size?: number;
  l2_size?: number;
  entity_count?: number;
}

export interface ClusterStats {
  mode: 'coordinator' | 'shard' | string;
  memory_count?: number;
  l1_size?: number;
  l2_size?: number;
  entity_count?: number;
  system?: {
    cpu?: number;
    ram?: { total: number; available: number; used: number; percent: number };
    disk?: { total: number; available: number; used: number; percent: number };
  };
  shards?: Record<string, ShardMetric>;
}

export interface ExportBundle {
  version?: string;
  server_version?: string;
  exported_at?: number;
  data_type?: string;
  tenant?: string;
  namespace?: string;
  counts?: {
    skills?: number;
    profiles?: number;
    memories?: number;
    triplets?: number;
  };
  data?: {
    skills?: unknown[];
    profiles?: ProfileItem[];
    memories?: unknown[];
    triplets?: Array<[string, string, string]>;
  };
}

export interface EntityMeta {
  type: NodeType;
  degree: number;
}

export type ViewMode = '2d' | '3d';
export type DockTab = 'explore' | 'recall' | 'skills' | 'profiles' | 'cluster';

export interface TypeaheadItem {
  label: string;
  value: string;
  type?: string;
  tag?: string;
}
