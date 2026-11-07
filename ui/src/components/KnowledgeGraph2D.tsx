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
  fit: () => void;
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

/**
 * Wraps long node labels at natural word boundaries so they fit within compact box badges,
 * preventing wide horizontal text overlapping neighboring nodes and edges.
 */
function wrapLabel(text: string, maxLineLen = 22, maxLines = 3): string {
  if (!text) return '';
  const trimmed = text.trim();
  if (trimmed.length <= maxLineLen) return trimmed;

  const words = trimmed.split(/\s+/);
  if (words.length === 1) {
    return trimmed.length > maxLineLen ? trimmed.slice(0, maxLineLen - 1) + '…' : trimmed;
  }

  const lines: string[] = [];
  let cur = '';

  for (const w of words) {
    if ((cur + ' ' + w).trim().length <= maxLineLen) {
      cur = (cur + ' ' + w).trim();
    } else {
      if (cur) lines.push(cur);
      cur = w;
      if (lines.length >= maxLines - 1) break;
    }
  }
  if (cur && lines.length < maxLines) lines.push(cur);

  const totalLength = lines.join(' ').length;
  if (totalLength < trimmed.length) {
    lines[lines.length - 1] = lines[lines.length - 1].replace(/[.,;:!?]?$/, '…');
  }
  return lines.join('\n');
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
    const nodesDS = useRef<DataSet<any> | null>(null);
    const edgesDS = useRef<DataSet<any> | null>(null);
    const physicsRef = useRef(physicsEnabled);
    const showEdgeLabelsRef = useRef(showEdgeLabels);
    const onNodeSelectRef = useRef(onNodeSelect);
    const isDarkRef = useRef(darkMode || isDarkMode());

    useEffect(() => {
      onNodeSelectRef.current = onNodeSelect;
    }, [onNodeSelect]);

    useEffect(() => {
      physicsRef.current = physicsEnabled;
    }, [physicsEnabled]);

    useEffect(() => {
      showEdgeLabelsRef.current = showEdgeLabels;
    }, [showEdgeLabels]);

    useEffect(() => {
      isDarkRef.current = darkMode || isDarkMode();
    }, [darkMode]);

    // Highlights direct neighborhood of selected node, dimming irrelevant clutter
    const highlightNeighborhood = (selectedId: string | null) => {
      if (!networkRef.current || !nodesDS.current || !edgesDS.current) return;
      const isDark = isDarkRef.current;

      if (!selectedId) {
        // Reset all nodes and edges to default appearance
        const allNodes = nodesDS.current.get();
        const resetNodes = allNodes.map((n: any) => ({
          id: n.id,
          opacity: 1.0,
          color: {
            background: n._origBg,
            border: n._origBorder,
            highlight: { background: n._origBg, border: n._origBorder },
            hover: { background: n._origBg, border: isDark ? '#FFFFFF' : '#1C1C1F' },
          },
          font: {
            color: n._origFontColor,
            size: 11,
            face: 'Plus Jakarta Sans, Inter, system-ui, sans-serif',
          },
        }));
        nodesDS.current.update(resetNodes);

        const allEdges = edgesDS.current.get();
        const resetEdges = allEdges.map((e: any) => ({
          id: e.id,
          color: {
            color: e._origColor,
            highlight: isDark ? '#2DD4BF' : '#0D9488',
            hover: isDark ? '#2DD4BF' : '#0D9488',
          },
          font: {
            size: showEdgeLabelsRef.current ? 9 : 0,
          },
        }));
        edgesDS.current.update(resetEdges);
        return;
      }

      const connectedNodeIds = new Set(
        networkRef.current.getConnectedNodes(selectedId).map(String)
      );
      connectedNodeIds.add(selectedId);
      const connectedEdgeIds = new Set(
        networkRef.current.getConnectedEdges(selectedId).map(String)
      );

      // Dim non-connected nodes
      const allNodes = nodesDS.current.get();
      const updatedNodes = allNodes.map((n: any) => {
        const isConnected = connectedNodeIds.has(String(n.id));
        const isTarget = String(n.id) === selectedId;
        return {
          id: n.id,
          opacity: isConnected ? 1.0 : 0.16,
          color: {
            background: isConnected
              ? n._origBg
              : isDark
              ? 'rgba(15, 23, 42, 0.25)'
              : 'rgba(240, 240, 240, 0.4)',
            border: isConnected
              ? n._origBorder
              : isDark
              ? 'rgba(148, 163, 184, 0.12)'
              : 'rgba(200, 200, 200, 0.25)',
            highlight: { background: n._origBg, border: isTarget ? '#FFFFFF' : n._origBorder },
            hover: { background: n._origBg, border: isDark ? '#FFFFFF' : '#1C1C1F' },
          },
          font: {
            color: isConnected
              ? n._origFontColor
              : isDark
              ? 'rgba(255,255,255,0.2)'
              : 'rgba(0,0,0,0.2)',
            size: isConnected ? 11 : 10,
            face: 'Plus Jakarta Sans, Inter, system-ui, sans-serif',
          },
        };
      });
      nodesDS.current.update(updatedNodes);

      // Dim non-connected edges
      const allEdges = edgesDS.current.get();
      const updatedEdges = allEdges.map((e: any) => {
        const isConnected = connectedEdgeIds.has(String(e.id));
        return {
          id: e.id,
          color: {
            color: isConnected
              ? isDark
                ? '#2DD4BF'
                : '#0D9488'
              : isDark
              ? 'rgba(255,255,255,0.03)'
              : 'rgba(0,0,0,0.03)',
            highlight: isDark ? '#2DD4BF' : '#0D9488',
            hover: isDark ? '#2DD4BF' : '#0D9488',
          },
          font: {
            size: isConnected && showEdgeLabelsRef.current ? 9 : 0,
          },
        };
      });
      edgesDS.current.update(updatedEdges);
    };

    useImperativeHandle(ref, () => ({
      focusNode: (id: string) => {
        if (!networkRef.current) return;
        try {
          networkRef.current.selectNodes([id]);
          highlightNeighborhood(id);
          networkRef.current.focus(id, {
            scale: 1.35,
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
        showEdgeLabelsRef.current = show;
        networkRef.current?.setOptions({ edges: { font: { size: show ? 9 : 0 } } });
      },
      fit: () => {
        networkRef.current?.fit({
          animation: { duration: 500, easingFunction: 'easeInOutQuad' },
        });
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
          selectConnectedEdges: true,
          hoverConnectedEdges: true,
          multiselect: false,
        },
        layout: {
          improvedLayout: false,
          randomSeed: 42,
        },
        physics: {
          enabled: true,
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -220, // Strong repulsion prevents overlapping
            centralGravity: 0.006,        // Gentle centering prevents dense clumping
            springLength: 220,            // Ample length allows nodes and relation pills to breathe
            springConstant: 0.04,         // Softer spring for natural layout
            avoidOverlap: 1.0,            // Strict 100% collision bounding box repulsion
            damping: 0.5,                 // Smooth settling without jitter
          },
          stabilization: {
            enabled: true,
            iterations: 150,
            updateInterval: 25,
            fit: true,
          },
        },
        edges: {
          width: 1.5,
          selectionWidth: 2.5,
          hoverWidth: 2,
        },
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
        if (id != null) {
          onNodeSelectRef.current?.(String(id));
          highlightNeighborhood(String(id));
        }
      });

      network.on('deselectNode', () => {
        highlightNeighborhood(null);
      });

      network.on('click', (params) => {
        if (!params.nodes || params.nodes.length === 0) {
          highlightNeighborhood(null);
        }
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

    // Sync graph data into DataSets
    useEffect(() => {
      if (!nodesDS.current || !edgesDS.current || !networkRef.current) return;

      if (rawNodes.length === 0) {
        nodesDS.current.clear();
        edgesDS.current.clear();
        return;
      }

      const isDark = darkMode || isDarkMode();
      isDarkRef.current = isDark;

      const links = (edges || [])
        .map((e, idx) => {
          const from = edgeSource(e);
          const to = edgeTarget(e);
          const pred = e.predicate || e.label || '';
          const origColor = isDark ? 'rgba(148, 163, 184, 0.4)' : 'rgba(100, 116, 139, 0.45)';
          return {
            id: `e-${idx}-${from}-${to}`,
            from,
            to,
            label: pred,
            arrows: {
              to: { enabled: true, scaleFactor: 0.75, type: 'arrow' },
            },
            font: {
              color: isDark ? '#CBD5E1' : '#334155',
              size: showEdgeLabels ? 9 : 0,
              face: 'Plus Jakarta Sans, Inter, system-ui, sans-serif',
              strokeWidth: 3,
              strokeColor: isDark ? '#0F172A' : '#FFFFFF',
              background: isDark ? 'rgba(15, 23, 42, 0.88)' : 'rgba(255, 255, 255, 0.88)',
              align: 'horizontal' as const,
            },
            color: {
              color: origColor,
              highlight: isDark ? '#2DD4BF' : '#0D9488',
              hover: isDark ? '#2DD4BF' : '#0D9488',
            },
            // Curved edges prevent mutual/parallel relations from overlapping on single straight line
            smooth: {
              enabled: true,
              type: 'curvedCW',
              roundness: 0.16,
            },
            width: 1.5,
            selectionWidth: 2.5,
            hoverWidth: 2,
            _origColor: origColor,
          };
        })
        .filter((l) => rawNodes.includes(l.from) && rawNodes.includes(l.to));

      const nodes = rawNodes.map((name) => {
        const type = nodeTypes[name] || 'concept';
        const st = TYPE_STYLES[type] || TYPE_STYLES.concept;
        const col = typeColor(type, isDark);
        const d = degrees?.[name] || 0;
        const wrappedLabel = wrapLabel(name, 22, 3);
        const bg = isDark ? 'rgba(23, 27, 38, 0.95)' : '#FFFFFF';
        const fontColor = isDark ? '#F1F5F9' : '#0F172A';

        return {
          id: name,
          label: wrappedLabel,
          title: `Entity: ${name}\nType: ${st.label}\nConnections: ${d}`,
          // Box shape ensures the physics solver calculates true text bounds and strictly avoids overlap
          shape: 'box',
          shapeProperties: { borderRadius: 6 },
          margin: { top: 8, right: 12, bottom: 8, left: 12 },
          color: {
            background: bg,
            border: col,
            highlight: { background: isDark ? 'rgba(30, 41, 59, 0.98)' : '#F8FAFC', border: col },
            hover: { background: isDark ? 'rgba(30, 41, 59, 0.98)' : '#F8FAFC', border: isDark ? '#FFFFFF' : '#1C1C1F' },
          },
          font: {
            color: fontColor,
            size: 11,
            face: 'Plus Jakarta Sans, Inter, system-ui, sans-serif',
          },
          borderWidth: Math.min(2 + Math.floor(d / 4), 4),
          shadow: {
            enabled: true,
            color: isDark ? 'rgba(0,0,0,0.45)' : 'rgba(0,0,0,0.08)',
            size: 6,
            x: 0,
            y: 2,
          },
          _origBg: bg,
          _origBorder: col,
          _origFontColor: fontColor,
        };
      });

      // Full replace keeps ids stable and avoids partial-merge bugs with filters
      nodesDS.current.clear();
      edgesDS.current.clear();
      nodesDS.current.add(nodes);
      edgesDS.current.add(links);

      // Re-run stabilization when data changes
      networkRef.current.setOptions({
        physics: { enabled: true },
        edges: { font: { size: showEdgeLabels ? 9 : 0 } },
      });
      networkRef.current.stabilize(120);
    }, [rawNodes, edges, degrees, nodeTypes, showEdgeLabels, darkMode]);

    return (
      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
        <div ref={hostRef} className="graph-canvas" style={{ width: '100%', height: '100%' }} />

        {/* Floating Quick-Action Toolbar */}
        <div
          style={{
            position: 'absolute',
            top: 14,
            right: 14,
            display: 'flex',
            gap: 6,
            zIndex: 10,
            background: darkMode ? 'rgba(23, 27, 38, 0.85)' : 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(8px)',
            padding: '4px 6px',
            borderRadius: 8,
            border: darkMode ? '1px solid rgba(255, 255, 255, 0.12)' : '1px solid rgba(0, 0, 0, 0.1)',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          <button
            type="button"
            title="Fit graph to view"
            className="btn-ds btn-ds-secondary"
            style={{ padding: '3px 8px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}
            onClick={() => networkRef.current?.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } })}
          >
            <span>⛶</span> Fit
          </button>
          <button
            type="button"
            title="Zoom In"
            className="btn-ds btn-ds-secondary"
            style={{ padding: '3px 8px', fontSize: 12, fontWeight: 600 }}
            onClick={() => {
              if (!networkRef.current) return;
              const scale = networkRef.current.getScale();
              networkRef.current.moveTo({ scale: scale * 1.3, animation: { duration: 250, easingFunction: 'easeInOutQuad' } });
            }}
          >
            +
          </button>
          <button
            type="button"
            title="Zoom Out"
            className="btn-ds btn-ds-secondary"
            style={{ padding: '3px 8px', fontSize: 12, fontWeight: 600 }}
            onClick={() => {
              if (!networkRef.current) return;
              const scale = networkRef.current.getScale();
              networkRef.current.moveTo({ scale: scale / 1.3, animation: { duration: 250, easingFunction: 'easeInOutQuad' } });
            }}
          >
            −
          </button>
          <button
            type="button"
            title="Re-stabilize layout"
            className="btn-ds btn-ds-secondary"
            style={{ padding: '3px 8px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}
            onClick={() => {
              if (!networkRef.current) return;
              highlightNeighborhood(null);
              networkRef.current.setOptions({ physics: { enabled: true } });
              networkRef.current.stabilize(100);
            }}
          >
            <span>⟳</span> Relax
          </button>
        </div>

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
