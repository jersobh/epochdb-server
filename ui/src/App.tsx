import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TopNav } from './components/TopNav';
import { ExplorerDock } from './components/ExplorerDock';
import { KnowledgeGraph2D, type Graph2DHandle } from './components/KnowledgeGraph2D';
import { KnowledgeGraph3D, type Graph3DHandle } from './components/KnowledgeGraph3D';
import { MemoryInspector } from './components/MemoryInspector';
import { AddTripletModal } from './components/AddTripletModal';
import { ImportExportModal } from './components/ImportExportModal';
import { AuthModal } from './components/AuthModal';
import { SeedModal } from './components/SeedModal';
import { Toast } from './components/Toast';
import {
  classifyNodes,
  computeDegrees,
  edgeSource,
  edgeTarget,
  normPred,
  TYPE_STYLES,
  typeColor,
  type NodeType,
} from './lib/graph';
import {
  connectSSE,
  ensureDefaultToken,
  fetchEntityGraph,
  fetchHealthz,
  fetchSkills,
  fetchStats,
  getNamespace,
  getTenant,
  getToken,
  saveAuth,
  setAuthRequiredHandler,
} from './lib/api';
import type { ClusterStats, DockTab, EntityMeta, SkillItem, TypeaheadItem, ViewMode } from './types';
import type { RawEdge } from './lib/graph';

export default function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('2d');
  const [dockOpen, setDockOpen] = useState(true);
  const [dockTab, setDockTab] = useState<DockTab>('explore');
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('epochdb_theme') === 'dark');
  const [tenant, setTenant] = useState(getTenant);
  const [namespace, setNamespace] = useState(getNamespace);
  const [isConnected, setIsConnected] = useState(false);
  const [shardMode, setShardMode] = useState(false);
  const [clusterModeLabel, setClusterModeLabel] = useState('Coordinator');

  const [rawNodes, setRawNodes] = useState<string[]>([]);
  const [rawEdges, setRawEdges] = useState<RawEdge[]>([]);
  const [nodeTypes, setNodeTypes] = useState<Record<string, NodeType>>({});
  const [degrees, setDegrees] = useState<Record<string, number>>({});
  const [disabledTypes, setDisabledTypes] = useState<Set<string>>(new Set());
  const [knownEntities, setKnownEntities] = useState<Map<string, EntityMeta>>(new Map());
  const [knownPredicates, setKnownPredicates] = useState<string[]>([]);

  const [focusValue, setFocusValue] = useState('');
  const [filterValue, setFilterValue] = useState('');
  const [depth, setDepth] = useState(2);
  const [physicsEnabled, setPhysicsEnabled] = useState(false);
  const [showEdgeLabels, setShowEdgeLabels] = useState(true);
  const [loading, setLoading] = useState(false);

  const [skills, setSkills] = useState<SkillItem[]>([]);
  const [stats, setStats] = useState<ClusterStats | null>(null);

  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectNodeId, setInspectNodeId] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [ieOpen, setIeOpen] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [seedOpen, setSeedOpen] = useState(false);
  const [toast, setToast] = useState({ message: '', visible: false });

  const graph2dRef = useRef<Graph2DHandle>(null);
  const graph3dRef = useRef<Graph3DHandle>(null);
  const toastTimer = useRef<number | null>(null);
  const knownPredicatesRef = useRef<string[]>([]);

  const showToast = useCallback((message: string) => {
    setToast({ message, visible: true });
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast({ message: '', visible: false }), 2800);
  }, []);

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
    document.body.classList.toggle('light-mode', !darkMode);
    localStorage.setItem('epochdb_theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    setAuthRequiredHandler(() => setAuthOpen(true));
    return () => setAuthRequiredHandler(null);
  }, []);

  const skillNameSet = useMemo(() => {
    return new Set(
      skills.map((s) =>
        String(s.skill_name || s.metadata?.skill_name || s.id || '').toLowerCase()
      )
    );
  }, [skills]);

  const loadSkills = useCallback(async () => {
    try {
      setSkills(await fetchSkills());
    } catch {
      /* ignore */
    }
  }, []);

  const loadStatsData = useCallback(async () => {
    try {
      const s = await fetchStats();
      setStats(s);
    } catch {
      /* ignore */
    }
  }, []);

  const loadGraph = useCallback(
    async (entityFilter?: string, depthOverride?: number) => {
      setLoading(true);
      try {
        const filter = (entityFilter ?? filterValue).trim();
        const d = depthOverride ?? depth;
        const data = await fetchEntityGraph(filter || undefined, d);
        const names = data.nodes || [];
        const edges = data.edges || [];
        const types = classifyNodes(names, edges, skillNameSet);
        const degs = computeDegrees(names, edges);
        const preds = new Set<string>(knownPredicatesRef.current);
        edges.forEach((e) => {
          const p = normPred(e.predicate || e.label);
          if (p) preds.add(p);
        });
        const predList = Array.from(preds);
        knownPredicatesRef.current = predList;

        setRawNodes(names);
        setRawEdges(edges);
        setNodeTypes(types);
        setDegrees(degs);
        setKnownPredicates(predList);

        setKnownEntities((prev) => {
          const next = filter ? new Map(prev) : new Map<string, EntityMeta>();
          names.forEach((n) => next.set(n, { type: types[n], degree: degs[n] || 0 }));
          return next;
        });
        setIsConnected(true);
      } catch (err) {
        console.error('Graph load error:', err);
        setIsConnected(false);
      } finally {
        setLoading(false);
      }
    },
    [filterValue, depth, skillNameSet]
  );

  const initApp = useCallback(async () => {
    ensureDefaultToken();
    setTenant(getTenant());
    setNamespace(getNamespace());
    try {
      const health = await fetchHealthz();
      setClusterModeLabel(health.mode === 'coordinator' ? 'Coordinator Gateway' : 'Shard Node');
      setShardMode(health.mode === 'shard');
      await loadStatsData();
      await loadSkills();
      await loadGraph();
    } catch (e) {
      console.error('Init failure:', e);
    }
  }, [loadGraph, loadSkills, loadStatsData]);

  useEffect(() => {
    void initApp();
    const es = connectSSE({
      onUpdate: () => {
        void loadStatsData();
        void loadGraph();
        void loadSkills();
      },
      onStats: (data) => setStats(data),
    });
    return () => es.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const visibleNodes = useMemo(
    () => rawNodes.filter((n) => !disabledTypes.has(nodeTypes[n] || 'concept')),
    [rawNodes, disabledTypes, nodeTypes]
  );
  const visibleSet = useMemo(() => new Set(visibleNodes), [visibleNodes]);
  const visibleEdges = useMemo(
    () => rawEdges.filter((e) => visibleSet.has(edgeSource(e)) && visibleSet.has(edgeTarget(e))),
    [rawEdges, visibleSet]
  );

  const suggestEntities = useCallback(
    (q: string, limit = 8): TypeaheadItem[] => {
      const ql = (q || '').trim().toLowerCase();
      const byDegree = (a: [string, EntityMeta], b: [string, EntityMeta]) =>
        b[1].degree - a[1].degree;
      const all = Array.from(knownEntities.entries());
      let res: Array<[string, EntityMeta]>;
      if (!ql) {
        res = all.sort(byDegree);
      } else {
        const exact: Array<[string, EntityMeta]> = [];
        const pre: Array<[string, EntityMeta]> = [];
        const sub: Array<[string, EntityMeta]> = [];
        all.forEach(([n, m]) => {
          const l = n.toLowerCase();
          if (l === ql) exact.push([n, m]);
          else if (l.startsWith(ql) || l.split(/[\s:_\-]+/).some((w) => w.startsWith(ql)))
            pre.push([n, m]);
          else if (l.includes(ql)) sub.push([n, m]);
        });
        res = exact.concat(pre.sort(byDegree), sub.sort(byDegree));
      }
      return res.slice(0, limit).map(([n, m]) => ({
        label: n,
        value: n,
        type: m.type,
        tag: (TYPE_STYLES[m.type] || TYPE_STYLES.concept).label,
      }));
    },
    [knownEntities]
  );

  const suggestRecall = useCallback(
    (value: string): TypeaheadItem[] => {
      const q = value || '';
      const ql = q.trim().toLowerCase();
      const out: TypeaheadItem[] = [];
      let hist: string[] = [];
      try {
        hist = JSON.parse(localStorage.getItem('epochdb_recall_history') || '[]');
      } catch {
        hist = [];
      }
      hist
        .filter((h) => h.toLowerCase() !== ql && (!ql || h.toLowerCase().includes(ql)))
        .slice(0, 3)
        .forEach((h) => out.push({ label: h, value: h, tag: 'recent' }));

      let frag = q.trim();
      let ents = suggestEntities(frag, 5);
      if (frag && !ents.length) {
        const m = q.match(/(\S+)$/);
        if (m && m[1].length >= 2) {
          frag = m[1];
          ents = suggestEntities(frag, 5);
        }
      }
      const prefix = frag ? q.slice(0, q.lastIndexOf(frag)) : '';
      ents.forEach((e) => out.push({ ...e, value: prefix + e.value }));

      if (ql) {
        skills.forEach((s) => {
          const name = String(s.skill_name || s.metadata?.skill_name || s.id || '');
          if (name && name.toLowerCase().includes(ql)) {
            out.push({ label: name, value: name, tag: 'skill', type: 'skill' });
          }
        });
      }
      const seen = new Set<string>();
      return out.filter((o) => !seen.has(o.value) && seen.add(o.value));
    },
    [suggestEntities, skills]
  );

  const focusNode = useCallback(
    async (nameRaw: string) => {
      const name = (nameRaw || '').trim();
      if (!name) return;
      let match = Array.from(knownEntities.keys()).find((n) => n.toLowerCase() === name.toLowerCase());
      if (!match) {
        const hits = suggestEntities(name, 1);
        if (hits.length) match = hits[0].value;
      }
      if (!match) {
        showToast('No matching node');
        return;
      }
      const matchType = nodeTypes[match] || 'concept';
      if (disabledTypes.has(matchType)) {
        setDisabledTypes((prev) => {
          const next = new Set(prev);
          next.delete(matchType);
          return next;
        });
      }
      if (!rawNodes.includes(match)) {
        setFilterValue(match);
        await loadGraph(match);
      }
      if (viewMode === '2d') graph2dRef.current?.focusNode(match);
      else graph3dRef.current?.focusNode(match);
    },
    [knownEntities, suggestEntities, nodeTypes, disabledTypes, rawNodes, loadGraph, viewMode, showToast]
  );

  const inspectNode = useCallback((nodeId: string) => {
    setInspectNodeId(nodeId);
    setInspectorOpen(true);
  }, []);

  const legendCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    rawNodes.forEach((n) => {
      const t = nodeTypes[n] || 'concept';
      counts[t] = (counts[t] || 0) + 1;
    });
    return counts;
  }, [rawNodes, nodeTypes]);

  return (
    <div className="studio-app">
      <TopNav
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        onToggleDock={() => setDockOpen((o) => !o)}
        dockOpen={dockOpen}
        onOpenImportExport={() => setIeOpen(true)}
        onOpenAddTriplet={() => setAddOpen(true)}
        onToggleTheme={() => {
          setDarkMode((d) => !d);
          void loadGraph();
        }}
        onOpenAuth={() => setAuthOpen(true)}
        tenant={tenant}
        namespace={namespace}
        isConnected={isConnected}
        nodeCount={visibleNodes.length}
        edgeCount={visibleEdges.length}
        darkMode={darkMode}
      />

      <div className={`shard-alert-banner${shardMode ? ' visible' : ''}`}>
        <strong>Internal Shard Storage Node</strong>: You are viewing an internal shard directly. Full
        cluster telemetry and global knowledge graph are served by the{' '}
        <strong>Coordinator Gateway</strong> at{' '}
        <a href="http://localhost:8080/visualize">http://localhost:8080/visualize</a>.
      </div>

      <main className="workspace">
        {/* Keep graph components mounted always — unmounting while vis/3d-force own
            DOM children causes removeChild NotFoundError (e.g. type filter empty→one). */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            // Prefer visibility over display:none so sibling 3D canvas keeps layout size
            visibility: viewMode === '2d' ? 'visible' : 'hidden',
            pointerEvents: viewMode === '2d' ? 'auto' : 'none',
            zIndex: viewMode === '2d' ? 1 : 0,
          }}
        >
          <KnowledgeGraph2D
            ref={graph2dRef}
            nodes={visibleNodes}
            edges={visibleEdges}
            degrees={degrees}
            nodeTypes={nodeTypes}
            physicsEnabled={physicsEnabled}
            showEdgeLabels={showEdgeLabels}
            onNodeSelect={inspectNode}
            darkMode={darkMode}
          />
          {loading ? (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)',
                fontSize: 13,
                background: 'color-mix(in srgb, var(--bg-app) 70%, transparent)',
                pointerEvents: 'none',
                zIndex: 5,
              }}
            >
              Traversing knowledge graph mesh...
            </div>
          ) : null}
        </div>

        <div
          style={{
            position: 'absolute',
            inset: 0,
            visibility: viewMode === '3d' ? 'visible' : 'hidden',
            pointerEvents: viewMode === '3d' ? 'auto' : 'none',
            zIndex: viewMode === '3d' ? 1 : 0,
          }}
        >
          <KnowledgeGraph3D
            ref={graph3dRef}
            nodes={visibleNodes}
            edges={visibleEdges}
            degrees={degrees}
            nodeTypes={nodeTypes}
            onNodeSelect={inspectNode}
            darkMode={darkMode}
            visible={viewMode === '3d'}
          />
        </div>

        <ExplorerDock
          open={dockOpen}
          onClose={() => setDockOpen(false)}
          activeTab={dockTab}
          onTabChange={setDockTab}
          focusValue={focusValue}
          onFocusChange={setFocusValue}
          onFocusSubmit={(v) => void focusNode(v)}
          filterValue={filterValue}
          onFilterChange={setFilterValue}
          onFilterSubmit={() => void loadGraph()}
          depth={depth}
          onDepthChange={(d) => {
            setDepth(d);
            void loadGraph(undefined, d);
          }}
          physicsEnabled={physicsEnabled}
          onPhysicsChange={(v) => {
            setPhysicsEnabled(v);
            graph2dRef.current?.setPhysics(v);
          }}
          showEdgeLabels={showEdgeLabels}
          onEdgeLabelsChange={(v) => {
            setShowEdgeLabels(v);
            graph2dRef.current?.setEdgeLabels(v);
          }}
          nodeTypes={nodeTypes}
          allNodeNames={rawNodes}
          disabledTypes={disabledTypes}
          onToggleType={(t, checked) => {
            setDisabledTypes((prev) => {
              const next = new Set(prev);
              if (checked === undefined) {
                if (next.has(t)) next.delete(t);
                else next.add(t);
              } else if (checked) next.delete(t);
              else next.add(t);
              return next;
            });
          }}
          onSetAllTypes={(enable) => {
            if (enable) setDisabledTypes(new Set());
            else setDisabledTypes(new Set(Object.keys(legendCounts)));
          }}
          onRefresh={() => void loadGraph()}
          onOpenSeed={() => setSeedOpen(true)}
          suggestEntities={suggestEntities}
          suggestRecall={suggestRecall}
          skills={skills}
          onLocateSkill={(name) => void focusNode(name)}
          stats={stats}
          clusterModeLabel={clusterModeLabel}
          darkMode={darkMode}
          onToast={showToast}
        />

        <div className="bottom-legend-bar">
          <div className="legend-chips">
            {Object.keys(legendCounts)
              .sort()
              .map((t) => {
                const st = TYPE_STYLES[t as NodeType];
                const isOff = disabledTypes.has(t);
                const radius =
                  st.shape === 'dot' || st.shape === 'ellipse'
                    ? '50%'
                    : st.shape === 'square' || st.shape === 'box'
                      ? '2px'
                      : '1px';
                return (
                  <div
                    key={t}
                    className="legend-chip"
                    style={{
                      opacity: isOff ? 0.35 : 1,
                      filter: isOff ? 'grayscale(0.85)' : undefined,
                      textDecoration: isOff ? 'line-through' : undefined,
                    }}
                    onClick={() => {
                      setDisabledTypes((prev) => {
                        const next = new Set(prev);
                        if (next.has(t)) next.delete(t);
                        else next.add(t);
                        return next;
                      });
                    }}
                    title={`Click to toggle filter for ${st.label}`}
                  >
                    <span
                      className="legend-color-dot"
                      style={{
                        background: typeColor(t, darkMode),
                        borderRadius: radius,
                        transform: st.shape === 'diamond' ? 'rotate(45deg)' : undefined,
                      }}
                    />
                    <span>{st.label}</span>
                    <span style={{ opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
                      {legendCounts[t]}
                    </span>
                  </div>
                );
              })}
          </div>
          <div className="legend-stats-badge">
            {visibleNodes.length} nodes · {visibleEdges.length} edges
          </div>
        </div>
      </main>

      <MemoryInspector
        open={inspectorOpen}
        nodeId={inspectNodeId}
        nodeType={inspectNodeId ? nodeTypes[inspectNodeId] || 'concept' : 'concept'}
        edges={rawEdges}
        skills={skills}
        onClose={() => setInspectorOpen(false)}
      />

      <AddTripletModal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onSuccess={() => {
          void loadStatsData();
          void loadGraph();
        }}
        onToast={showToast}
        suggestEntities={suggestEntities}
        knownPredicates={knownPredicates}
      />

      <ImportExportModal
        open={ieOpen}
        onClose={() => setIeOpen(false)}
        onToast={showToast}
        onImported={() => {
          void loadStatsData();
          void loadGraph();
        }}
      />

      <AuthModal
        open={authOpen}
        onClose={() => setAuthOpen(false)}
        initialToken={getToken()}
        initialTenant={tenant}
        initialNamespace={namespace}
        onSave={(token, t, ns) => {
          saveAuth(token, t, ns);
          setTenant(t);
          setNamespace(ns);
          setAuthOpen(false);
          showToast('Scope and credentials saved!');
          void initApp();
        }}
      />

      <SeedModal
        open={seedOpen}
        onClose={() => setSeedOpen(false)}
        tenant={tenant}
        namespace={namespace}
        onDone={(msg) => {
          showToast(msg);
          void loadStatsData();
          void loadSkills();
          void loadGraph();
        }}
      />

      <Toast message={toast.message} visible={toast.visible} />
    </div>
  );
}
