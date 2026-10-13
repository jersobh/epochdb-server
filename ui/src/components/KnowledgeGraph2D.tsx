import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { Network, DataSet } from 'vis-network/standalone';
import type { Options } from 'vis-network';
import type { RawEdge } from '../lib/graph';
import {
  TYPE_STYLES,
  edgeSource,
  edgeTarget,
  isDarkMode,
  typeColor,
  type NodeType,
} from '../lib/graph';

export interface Graph2DHandle {
  focusNode: (id: string) => void;
  setPhysics: (enabled: boolean) => void;
  setEdgeLabels: (show: boolean) => void;
}

interface KnowledgeGraph2DProps {
  nodes: string[];
  edges: RawEdge[];
  degrees: Record<string, number>;
  nodeTypes: Record<string, NodeType>;
  physicsEnabled: boolean;
  showEdgeLabels: boolean;
  onNodeSelect?: (nodeId: string) => void;
  darkMode: boolean;
}

export const KnowledgeGraph2D = forwardRef<Graph2DHandle, KnowledgeGraph2DProps>(
  function KnowledgeGraph2D(
    {
      nodes: rawNodes,
      edges,
      degrees,
      nodeTypes,
      physicsEnabled,
      showEdgeLabels,
      onNodeSelect,
      darkMode,
    },
    ref
  ) {
    // React owns hostRef; vis-network exclusively owns hostRef's child mountEl.
    const hostRef = useRef<HTMLDivElement>(null);
    const mountRef = useRef<HTMLDivElement | null>(null);
    const networkRef = useRef<Network | null>(null);
    const nodesDS = useRef<DataSet<object> | null>(null);
    const edgesDS = useRef<DataSet<object> | null>(null);
    const physicsRef = useRef(physicsEnabled);
    const onNodeSelectRef = useRef(onNodeSelect);

    useEffect(() => {
      onNodeSelectRef.current = onNodeSelect;
    }, [onNodeSelect]);

    useEffect(() => {
      physicsRef.current = physicsEnabled;
    }, [physicsEnabled]);

    useImperativeHandle(ref, () => ({
      focusNode: (id: string) => {
        if (!networkRef.current) return;
        try {
          networkRef.current.selectNodes([id]);
          networkRef.current.focus(id, {
            scale: 1.5,
            animation: { duration: 600, easingFunction: 'easeInOutQuad' },
          });
        } catch {
          /* node may not be in current view */
        }
      },
      setPhysics: (enabled: boolean) => {
        physicsRef.current = enabled;
        networkRef.current?.setOptions({ physics: { enabled } });
      },
      setEdgeLabels: (show: boolean) => {
        networkRef.current?.setOptions({ edges: { font: { size: show ? 10 : 0 } } });
      },
    }));

    // Create stable mount + Network once
    useEffect(() => {
      const host = hostRef.current;
      if (!host) return;

      const mountEl = document.createElement('div');
      mountEl.style.width = '100%';
      mountEl.style.height = '100%';
      host.appendChild(mountEl);
      mountRef.current = mountEl;

      nodesDS.current = new DataSet();
      edgesDS.current = new DataSet();

      const options: Options = {
        autoResize: true,
        interaction: {
          hover: true,
          tooltipDelay: 100,
          zoomView: true,
          dragView: true,
          selectable: true,
        },
        layout: {
          // Avoid vis-network "improved layout" failures on larger graphs
          improvedLayout: false,
          randomSeed: 42,
        },
        physics: {
          enabled: true,
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -42,
            centralGravity: 0.012,
            springLength: 110,
            springConstant: 0.06,
            avoidOverlap: 0.6,
          },
          stabilization: { iterations: 100, fit: true },
        },
        edges: { width: 1.3 },
      };

      const network = new Network(
        mountEl,
        { nodes: nodesDS.current, edges: edgesDS.current },
        options
      );
      networkRef.current = network;

      network.on('stabilizationIterationsDone', () => {
        if (!physicsRef.current) {
          network.setOptions({ physics: { enabled: false } });
        }
      });

      network.on('selectNode', (params) => {
        const id = params.nodes?.[0];
        if (id != null) onNodeSelectRef.current?.(String(id));
      });

      return () => {
        try {
          network.destroy();
        } catch {
          /* already torn down */
        }
        networkRef.current = null;
        nodesDS.current = null;
        edgesDS.current = null;
        if (mountRef.current && mountRef.current.parentNode === host) {
          host.removeChild(mountRef.current);
        }
        mountRef.current = null;
      };
    }, []);

    // Sync graph data into DataSets (no React DOM mutation)
    useEffect(() => {
      if (!nodesDS.current || !edgesDS.current || !networkRef.current) return;

      if (rawNodes.length === 0) {
        nodesDS.current.clear();
        edgesDS.current.clear();
        return;
      }

      const isDark = darkMode || isDarkMode();
      const links = (edges || [])
        .map((e, idx) => ({
          id: `e-${idx}-${edgeSource(e)}-${edgeTarget(e)}`,
          from: edgeSource(e),
          to: edgeTarget(e),
          label: e.predicate || e.label || '',
          arrows: 'to',
          font: {
            color: isDark ? '#9CA3AF' : '#52525B',
            size: showEdgeLabels ? 10 : 0,
            face: 'IBM Plex Sans',
            strokeWidth: 2,
            strokeColor: isDark ? '#14171F' : '#FAFAF9',
            align: 'middle' as const,
          },
          color: {
            color: isDark ? 'rgba(255,255,255,0.18)' : '#B8B8B0',
            highlight: isDark ? '#14B8A6' : '#0F766E',
            hover: isDark ? '#14B8A6' : '#0F766E',
          },
          smooth: false,
        }))
        .filter((l) => rawNodes.includes(l.from) && rawNodes.includes(l.to));

      const labelColor = isDark ? '#E5E7EB' : '#1C1C1F';
      const nodes = rawNodes.map((name) => {
        const type = nodeTypes[name] || 'concept';
        const st = TYPE_STYLES[type] || TYPE_STYLES.concept;
        const col = typeColor(type, isDark);
        const inside = st.shape === 'box' || st.shape === 'ellipse';
        const d = degrees?.[name] || 0;
        return {
          id: name,
          label: name,
          title: `${name} · ${st.label}`,
          shape: st.shape,
          size: 11 + Math.min(d, 8) * 1.6,
          color: {
            background: col,
            border: col,
            highlight: { background: col, border: isDark ? '#FFFFFF' : '#1C1C1F' },
            hover: { background: col, border: isDark ? '#FFFFFF' : '#1C1C1F' },
          },
          font: {
            color: inside ? '#FFFFFF' : labelColor,
            size: 11,
            face: 'Plus Jakarta Sans',
          },
          borderWidth: 2,
          shadow: { enabled: true, color: 'rgba(0,0,0,0.15)', size: 4, x: 1, y: 1 },
        };
      });

      // Full replace keeps ids stable and avoids partial-merge bugs with filters
      nodesDS.current.clear();
      edgesDS.current.clear();
      nodesDS.current.add(nodes);
      edgesDS.current.add(links);

      // Re-run a short stabilization when data changes
      networkRef.current.setOptions({
        physics: { enabled: true },
        edges: { font: { size: showEdgeLabels ? 10 : 0 } },
      });
      networkRef.current.stabilize(80);
    }, [rawNodes, edges, degrees, nodeTypes, showEdgeLabels, darkMode]);

    return (
      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
        <div ref={hostRef} className="graph-canvas" style={{ width: '100%', height: '100%' }} />
        {rawNodes.length === 0 ? (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 10,
              color: 'var(--text-muted)',
              fontSize: 13,
              padding: 20,
              textAlign: 'center',
              pointerEvents: 'none',
              background: 'var(--bg-app)',
            }}
          >
            <div style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
              No triples in current scope
            </div>
            <div style={{ fontSize: 12, maxWidth: 360 }}>
              Add a triplet via the "+ Add Triplet" button or import procedural skills to construct
              the neuro-symbolic knowledge graph.
            </div>
          </div>
        ) : null}
      </div>
    );
  }
);
