import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  Chart,
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
} from 'chart.js';
import { TypeaheadInput } from './TypeaheadInput';
import { TYPE_STYLES, typeColor, type NodeType } from '../lib/graph';
import { fetchProfiles, recallQuery } from '../lib/api';
import type { ClusterStats, DockTab, SkillItem, TypeaheadItem } from '../types';

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Legend, Tooltip);

interface ExplorerDockProps {
  open: boolean;
  onClose: () => void;
  activeTab: DockTab;
  onTabChange: (tab: DockTab) => void;
  focusValue: string;
  onFocusChange: (v: string) => void;
  onFocusSubmit: (v: string) => void;
  filterValue: string;
  onFilterChange: (v: string) => void;
  onFilterSubmit: () => void;
  depth: number;
  onDepthChange: (d: number) => void;
  physicsEnabled: boolean;
  onPhysicsChange: (v: boolean) => void;
  showEdgeLabels: boolean;
  onEdgeLabelsChange: (v: boolean) => void;
  nodeTypes: Record<string, NodeType>;
  allNodeNames: string[];
  disabledTypes: Set<string>;
  onToggleType: (t: string, checked?: boolean) => void;
  onSetAllTypes: (enable: boolean) => void;
  onRefresh: () => void;
  onOpenSeed: () => void;
  suggestEntities: (q: string) => TypeaheadItem[];
  suggestRecall: (q: string) => TypeaheadItem[];
  skills: SkillItem[];
  onLocateSkill: (name: string) => void;
  stats: ClusterStats | null;
  clusterModeLabel: string;
  darkMode: boolean;
  onToast: (msg: string) => void;
}

function skillName(s: SkillItem): string {
  return String(s.skill_name || s.metadata?.skill_name || s.id || '');
}

function barRow(label: string, pctRaw: number | undefined) {
  const pct = Math.max(0, Math.min(100, Math.round(pctRaw || 0)));
  const c =
    pct > 85 ? 'var(--accent-rose)' : pct > 60 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
  return (
    <div className="bar-row" key={label}>
      <span>{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: c }} />
      </div>
      <span className="bar-val">{pct}%</span>
    </div>
  );
}

export function ExplorerDock({
  open,
  onClose,
  activeTab,
  onTabChange,
  focusValue,
  onFocusChange,
  onFocusSubmit,
  filterValue,
  onFilterChange,
  onFilterSubmit,
  depth,
  onDepthChange,
  physicsEnabled,
  onPhysicsChange,
  showEdgeLabels,
  onEdgeLabelsChange,
  nodeTypes,
  allNodeNames,
  disabledTypes,
  onToggleType,
  onSetAllTypes,
  onRefresh,
  onOpenSeed: _onOpenSeed,
  suggestEntities,
  suggestRecall,
  skills,
  onLocateSkill,
  stats,
  clusterModeLabel,
  darkMode,
  onToast,
}: ExplorerDockProps) {
  const [skillsFilter, setSkillsFilter] = useState('');
  const [profilesFilter, setProfilesFilter] = useState('');
  const [profiles, setProfiles] = useState<Array<{ user_id?: string; payload?: string; text?: string; metadata?: Record<string, unknown> }>>([]);
  const [recallQueryText, setRecallQueryText] = useState('');
  const [recallK, setRecallK] = useState(5);
  const [recallResults, setRecallResults] = useState<ReactNode>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  const typeCounts: Record<string, number> = {};
  allNodeNames.forEach((n) => {
    const t = nodeTypes[n] || 'entity';
    typeCounts[t] = (typeCounts[t] || 0) + 1;
  });
  // Discovered entity types derived dynamically from active graph nodes:
  const orderedTypes = Object.keys(typeCounts).sort(
    (a, b) => (typeCounts[b] || 0) - (typeCounts[a] || 0)
  );

  const loadProfiles = async () => {
    try {
      const list = await fetchProfiles();
      setProfiles(list);
    } catch (e) {
      onToast(e instanceof Error ? e.message : 'Failed to load profiles');
    }
  };

  useEffect(() => {
    if (activeTab === 'profiles') void loadProfiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== 'cluster' || !chartRef.current || !stats) return;

    const labels: string[] = [];
    const cpuData: number[] = [];
    const ramData: number[] = [];
    const diskData: number[] = [];

    if (stats.mode === 'coordinator' && stats.shards) {
      if (stats.system) {
        labels.push('Coordinator');
        cpuData.push(stats.system.cpu || 0);
        ramData.push(stats.system.ram?.percent || 0);
        diskData.push(stats.system.disk?.percent || 0);
      }
      for (const [shardUrl, sh] of Object.entries(stats.shards)) {
        const name = shardUrl.replace('http://', '').replace('https://', '').replace(':8080', '');
        labels.push(name);
        cpuData.push(sh.cpu || 0);
        ramData.push(sh.ram?.percent || 0);
        diskData.push(sh.disk?.percent || 0);
      }
    } else {
      const sys = stats.system || {};
      labels.push('Local Node');
      cpuData.push(sys.cpu || 0);
      ramData.push(sys.ram?.percent || 0);
      diskData.push(sys.disk?.percent || 0);
    }

    const grid = darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.07)';
    const text = darkMode ? '#D1D5DB' : '#3F3F46';

    if (chartInstance.current) {
      chartInstance.current.data.labels = labels;
      chartInstance.current.data.datasets[0].data = cpuData;
      chartInstance.current.data.datasets[1].data = ramData;
      chartInstance.current.data.datasets[2].data = diskData;
      const opts = chartInstance.current.options;
      if (opts.scales?.x?.grid) opts.scales.x.grid.color = grid;
      if (opts.scales?.y?.grid) opts.scales.y.grid.color = grid;
      if (opts.scales?.x?.ticks) opts.scales.x.ticks.color = text;
      if (opts.scales?.y?.ticks) opts.scales.y.ticks.color = text;
      if (opts.plugins?.legend?.labels) opts.plugins.legend.labels.color = text;
      chartInstance.current.update('none');
      chartInstance.current.resize();
      return;
    }

    chartInstance.current = new Chart(chartRef.current.getContext('2d')!, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'CPU %',
            data: cpuData,
            backgroundColor: 'rgba(249, 115, 22, 0.75)',
            borderColor: '#f97316',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'RAM %',
            data: ramData,
            backgroundColor: 'rgba(34, 197, 94, 0.75)',
            borderColor: '#22c55e',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'Disk %',
            data: diskData,
            backgroundColor: 'rgba(59, 130, 246, 0.75)',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 }, color: text } },
        },
        scales: {
          x: { grid: { color: grid }, ticks: { font: { size: 9 }, color: text } },
          y: { min: 0, max: 100, grid: { color: grid }, ticks: { font: { size: 9 }, color: text } },
        },
      },
    });
  }, [activeTab, stats, darkMode, open]);

  const executeRecall = async () => {
    const q = recallQueryText.trim();
    if (!q) return;
    try {
      const hist = JSON.parse(localStorage.getItem('epochdb_recall_history') || '[]') as string[];
      localStorage.setItem(
        'epochdb_recall_history',
        JSON.stringify([q, ...hist.filter((x) => x !== q)].slice(0, 20))
      );
    } catch {
      /* ignore */
    }
    setRecallResults(<div style={{ color: 'var(--text-muted)' }}>Searching embeddings...</div>);
    try {
      const res = await recallQuery(q, Math.max(1, Math.min(100, recallK || 5)));
      const mems = res.memories || [];
      if (!mems.length) {
        setRecallResults(<div style={{ color: 'var(--text-muted)' }}>No matching memories found.</div>);
        return;
      }
      setRecallResults(
        <>
          {mems.map((m, i) => (
            <div key={i} className="metric-card">
              <div style={{ fontWeight: 600 }}>{m.payload || m.text}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                score: {Number(m.score || m.similarity || 1).toFixed(3)}
              </div>
            </div>
          ))}
        </>
      );
    } catch (err) {
      setRecallResults(
        <div style={{ color: 'var(--accent-rose)' }}>
          {err instanceof Error ? err.message : String(err)}
        </div>
      );
    }
  };

  const filteredSkills = skills.filter((s) => {
    const q = skillsFilter.trim().toLowerCase();
    const hay = `${skillName(s)} ${s.description || s.metadata?.description || ''}`.toLowerCase();
    return !q || hay.includes(q);
  });

  const profileGroups: Record<string, string[]> = {};
  profiles.forEach((p) => {
    const uid = String(p.metadata?.user_id || p.user_id || 'unknown');
    const fact = String(p.payload || p.text || p.metadata?.fact_text || '');
    const q = profilesFilter.trim().toLowerCase();
    if (q && !`${uid} ${fact}`.toLowerCase().includes(q)) return;
    (profileGroups[uid] = profileGroups[uid] || []).push(fact);
  });
  const profileUsers = Object.keys(profileGroups).sort();

  const shardCards: ReactNode[] = [];
  if (stats) {
    if (stats.mode === 'coordinator' && stats.shards) {
      if (stats.system) {
        shardCards.push(
          <div className="metric-card" key="coord">
            <div className="metric-title">
              <span>Coordinator Gateway</span>
              <span style={{ color: 'var(--accent-emerald)', fontSize: 11 }}>online</span>
            </div>
            {barRow('CPU', stats.system.cpu)}
            {barRow('RAM', stats.system.ram?.percent)}
            {barRow('Disk', stats.system.disk?.percent)}
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {stats.memory_count || 0} memories · {stats.entity_count || 0} entities
            </div>
          </div>
        );
      }
      for (const [shardUrl, sh] of Object.entries(stats.shards)) {
        const name = shardUrl.replace('http://', '').replace('https://', '').replace(':8080', '');
        const online = sh.status === 'healthy';
        shardCards.push(
          <div className="metric-card" key={shardUrl}>
            <div className="metric-title">
              <span>{name}</span>
              <span style={{ color: online ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontSize: 11 }}>
                {online ? 'online' : 'offline'}
              </span>
            </div>
            {online ? (
              <>
                {barRow('CPU', sh.cpu)}
                {barRow('RAM', sh.ram?.percent)}
                {barRow('Disk', sh.disk?.percent)}
              </>
            ) : null}
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {sh.memory_count || 0} memories · L1 {sh.l1_size || 0} · L2 {sh.l2_size || 0}
            </div>
          </div>
        );
      }
    } else {
      const sys = stats.system || {};
      shardCards.push(
        <div className="metric-card" key="local">
          <div className="metric-title">
            <span>Local Shard Node</span>
            <span style={{ color: 'var(--accent-emerald)', fontSize: 11 }}>online</span>
          </div>
          {barRow('CPU', sys.cpu)}
          {barRow('RAM', sys.ram?.percent)}
          {barRow('Disk', sys.disk?.percent)}
          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            {stats.memory_count || 0} memories stored
          </div>
        </div>
      );
    }
  }

  return (
    <aside className={`floating-dock${open ? '' : ' collapsed'}`}>
      <div className="dock-header">
        <span style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-sans)' }}>
          Explorer & Telemetry
        </span>
        <button type="button" className="close-btn" onClick={onClose}>
          &times;
        </button>
      </div>

      <div className="dock-tabs">
        {(
          [
            ['explore', 'Explore'],
            ['recall', 'Recall'],
            ['skills', 'Skills'],
            ['profiles', 'Profiles'],
            ['cluster', 'Cluster'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={`dock-tab${activeTab === id ? ' active' : ''}`}
            onClick={() => onTabChange(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="dock-content">
        {activeTab === 'explore' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="input-group">
              <label className="input-label">Search &amp; Focus</label>
              <TypeaheadInput
                placeholder="Find a node and zoom to it..."
                value={focusValue}
                onChange={onFocusChange}
                getItems={suggestEntities}
                onPick={(it) => onFocusSubmit(it.value)}
                onSubmit={onFocusSubmit}
              />
            </div>
            <div className="input-group">
              <label className="input-label">Filter Entities</label>
              <TypeaheadInput
                placeholder="Show only the subgraph around an entity..."
                value={filterValue}
                onChange={(v) => {
                  onFilterChange(v);
                  if (!v.trim()) {
                    setTimeout(() => onFilterSubmit(), 250);
                  }
                }}
                getItems={suggestEntities}
                onPick={() => onFilterSubmit()}
                onSubmit={() => onFilterSubmit()}
              />
            </div>
            <div className="input-group">
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <label className="input-label">Traversal Depth</label>
                <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  {depth} hops
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={10}
                value={depth}
                onChange={(e) => onDepthChange(Number(e.target.value))}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 6 }}>
              <span className="input-label">Physics Simulation</span>
              <input
                type="checkbox"
                checked={physicsEnabled}
                onChange={(e) => onPhysicsChange(e.target.checked)}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="input-label">Show Edge Labels</span>
              <input
                type="checkbox"
                checked={showEdgeLabels}
                onChange={(e) => onEdgeLabelsChange(e.target.checked)}
              />
            </div>
            <div className="input-group" style={{ paddingTop: 8, borderTop: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                <label className="input-label">Object / Entity Type</label>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button
                    type="button"
                    className="btn-ds btn-ds-secondary"
                    style={{ padding: '1px 7px', fontSize: 10 }}
                    onClick={() => onSetAllTypes(true)}
                  >
                    All
                  </button>
                  <button
                    type="button"
                    className="btn-ds btn-ds-secondary"
                    style={{ padding: '1px 7px', fontSize: 10 }}
                    onClick={() => onSetAllTypes(false)}
                  >
                    None
                  </button>
                </div>
              </div>
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 2,
                  maxHeight: 180,
                  overflowY: 'auto',
                  padding: 4,
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-app)',
                }}
              >
                {orderedTypes.length === 0 ? (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '12px 8px', textAlign: 'center', fontStyle: 'italic' }}>
                    No entity types discovered yet. Ingest knowledge or run extraction to discover entities dynamically.
                  </div>
                ) : (
                  orderedTypes.map((t) => {
                    const st = TYPE_STYLES[t as NodeType];
                    const cnt = typeCounts[t] || 0;
                    const radius =
                      st.shape === 'dot' || st.shape === 'ellipse'
                        ? '50%'
                        : st.shape === 'square' || st.shape === 'box'
                          ? '2px'
                          : '1px';
                    const rot = st.shape === 'diamond' ? 'transform: rotate(45deg)' : '';
                    return (
                      <label
                        key={t}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          padding: '4px 6px',
                          borderRadius: 'var(--radius-sm)',
                          cursor: 'pointer',
                          userSelect: 'none',
                          fontSize: 12,
                          opacity: cnt === 0 ? 0.45 : 1,
                        }}
                      >
                        <input
                          type="checkbox"
                          style={{ accentColor: 'var(--accent-teal)', cursor: 'pointer' }}
                          checked={!disabledTypes.has(t)}
                          onChange={(e) => onToggleType(t, e.target.checked)}
                        />
                        <span
                          style={{
                            width: 8,
                            height: 8,
                            background: typeColor(t),
                            borderRadius: radius,
                            flexShrink: 0,
                            display: 'inline-block',
                            ...(rot ? { transform: 'rotate(45deg)' } : {}),
                          }}
                        />
                        <span style={{ flex: 1, fontWeight: 500, color: 'var(--text-primary)' }}>{st.label}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                          {cnt}
                        </span>
                      </label>
                    );
                  })
                )}
              </div>
            </div>
            <div style={{ paddingTop: 8, borderTop: '1px solid var(--border-subtle)', display: 'flex', gap: 8 }}>
              <button
                type="button"
                className="btn-ds btn-ds-secondary"
                style={{ flex: 1, justifyContent: 'center' }}
                onClick={onRefresh}
              >
                Refresh Graph
              </button>
              {/* Seed Sample disabled — Percepta must not auto-load demo graphs */}
            </div>
          </div>
        )}

        {activeTab === 'recall' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="input-group">
              <label className="input-label">Semantic Vector Query</label>
              <TypeaheadInput
                placeholder="e.g. quantum theory, algorithm history"
                value={recallQueryText}
                onChange={setRecallQueryText}
                getItems={suggestRecall}
                onSubmit={() => void executeRecall()}
              />
            </div>
            <div className="input-group">
              <label className="input-label">Top K (results)</label>
              <input
                type="number"
                className="ds-input"
                min={1}
                max={100}
                value={recallK}
                onChange={(e) => setRecallK(Number(e.target.value))}
              />
            </div>
            <button
              type="button"
              className="btn-ds btn-ds-primary"
              style={{ justifyContent: 'center' }}
              onClick={() => void executeRecall()}
            >
              Search Memories
            </button>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12 }}>
              {recallResults}
            </div>
          </div>
        )}

        {activeTab === 'skills' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <input
              type="text"
              className="ds-input"
              placeholder="Filter skills..."
              value={skillsFilter}
              onChange={(e) => setSkillsFilter(e.target.value)}
              autoComplete="off"
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {!filteredSkills.length ? (
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {skillsFilter ? 'No skills match.' : 'No procedural skills registered yet.'}
                </div>
              ) : (
                filteredSkills.map((s, i) => {
                  const steps = (s.steps || s.metadata?.steps || []) as Array<string | { action?: string }>;
                  const desc = String(s.description || s.metadata?.description || '');
                  const name = skillName(s);
                  return (
                    <div key={`${name}-${i}`} className="metric-card">
                      <div className="metric-title">
                        <span>{name}</span>
                        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                          {steps.length} steps
                        </span>
                      </div>
                      {desc ? <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{desc}</div> : null}
                      {steps.length ? (
                        <ol
                          style={{
                            fontSize: 11,
                            color: 'var(--text-secondary)',
                            paddingLeft: 18,
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 2,
                          }}
                        >
                          {steps.map((st, n) => (
                            <li key={n}>
                              {typeof st === 'string' ? st : st.action || JSON.stringify(st)}
                            </li>
                          ))}
                        </ol>
                      ) : null}
                      <button
                        type="button"
                        className="btn-ds btn-ds-secondary"
                        style={{ padding: '3px 8px', fontSize: 11, alignSelf: 'flex-start' }}
                        onClick={() => onLocateSkill(name)}
                      >
                        Locate in graph
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {activeTab === 'profiles' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                type="text"
                className="ds-input"
                placeholder="Filter by user or fact..."
                value={profilesFilter}
                onChange={(e) => setProfilesFilter(e.target.value)}
                autoComplete="off"
              />
              <button type="button" className="btn-ds btn-ds-secondary" onClick={() => void loadProfiles()}>
                Refresh
              </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {!profileUsers.length ? (
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {profilesFilter ? 'No profiles match.' : 'No user profiles stored yet.'}
                </div>
              ) : (
                profileUsers.map((u) => (
                  <div key={u} className="metric-card">
                    <div className="metric-title">
                      <span>{u}</span>
                      <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        {profileGroups[u].length} facts
                      </span>
                    </div>
                    <ul
                      style={{
                        fontSize: 11,
                        color: 'var(--text-secondary)',
                        paddingLeft: 16,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                      }}
                    >
                      {profileGroups[u].map((f, i) => (
                        <li key={i}>{f}</li>
                      ))}
                    </ul>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'cluster' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, fontWeight: 600 }}>Node Shards Health</span>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>
                {clusterModeLabel}
              </span>
            </div>
            <div style={{ position: 'relative', height: 190, width: '100%' }}>
              <canvas ref={chartRef} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>{shardCards}</div>
          </div>
        )}
      </div>
    </aside>
  );
}
