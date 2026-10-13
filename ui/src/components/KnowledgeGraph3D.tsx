import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import ForceGraph3D from '3d-force-graph';
import type { RawEdge } from '../lib/graph';
import { TYPE_STYLES, edgeSource, edgeTarget, typeColor, type NodeType } from '../lib/graph';

export interface Graph3DHandle {
  focusNode: (id: string) => void;
}

interface KnowledgeGraph3DProps {
  nodes: string[];
  edges: RawEdge[];
  degrees: Record<string, number>;
  nodeTypes: Record<string, NodeType>;
  onNodeSelect?: (nodeId: string) => void;
  darkMode: boolean;
  visible: boolean;
}

export const KnowledgeGraph3D = forwardRef<Graph3DHandle, KnowledgeGraph3DProps>(
  function KnowledgeGraph3D(
    { nodes: rawNodes, edges, degrees, nodeTypes, onNodeSelect, darkMode, visible },
    ref
  ) {
    // React owns hostRef; ForceGraph3D owns a dedicated child mount element.
    // Never use display:none here — a 0×0 WebGL canvas breaks OrbitControls.
    const hostRef = useRef<HTMLDivElement>(null);
    const mountRef = useRef<HTMLDivElement | null>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const graphRef = useRef<any>(null);
    const onNodeSelectRef = useRef(onNodeSelect);
    const visibleRef = useRef(visible);

    useEffect(() => {
      onNodeSelectRef.current = onNodeSelect;
    }, [onNodeSelect]);

    useEffect(() => {
      visibleRef.current = visible;
    }, [visible]);

    useImperativeHandle(ref, () => ({
      focusNode: (id: string) => {
        const g = graphRef.current;
        if (!g) return;
        const n = g.graphData().nodes.find((x: { id: string }) => x.id === id);
        if (n && typeof n.x === 'number' && typeof n.y === 'number') {
          const dist = 90;
          const z = typeof n.z === 'number' ? n.z : 0;
          const r = 1 + dist / Math.hypot(n.x, n.y, z || 0.0001);
          g.cameraPosition({ x: n.x * r, y: n.y * r, z: z * r }, n, 1000);
        }
      },
    }));

    const syncSize = () => {
      const g = graphRef.current;
      const mount = mountRef.current;
      const host = hostRef.current;
      if (!g || !mount || !host) return;
      const w = host.clientWidth || host.offsetWidth;
      const h = host.clientHeight || host.offsetHeight;
      if (w > 0 && h > 0) {
        mount.style.width = `${w}px`;
        mount.style.height = `${h}px`;
        g.width(w);
        g.height(h);
        // Re-assert navigation after resize (controls can go stale at 0×0)
        try {
          g.enableNavigationControls(true);
          const controls = g.controls?.();
          if (controls) {
            controls.enabled = visibleRef.current;
            controls.update?.();
          }
        } catch {
          /* ignore */
        }
      }
    };

    // Create graph once against a stable child element
    useEffect(() => {
      const host = hostRef.current;
      if (!host) return;

      const mountEl = document.createElement('div');
      mountEl.style.width = '100%';
      mountEl.style.height = '100%';
      mountEl.style.touchAction = 'none';
      host.appendChild(mountEl);
      mountRef.current = mountEl;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const createGraph = ForceGraph3D as any;
      const graph = createGraph()(mountEl)
        .graphData({ nodes: [], links: [] })
        .backgroundColor('#F4F4F2')
        .nodeId('id')
        .linkSource('source')
        .linkTarget('target')
        .enableNavigationControls(true)
        .enableNodeDrag(true)
        .showNavInfo(false)
        .nodeLabel(
          (node: { id: string; type?: string }) =>
            `<div style="background: rgba(20,20,22,0.9); padding: 4px 8px; border-radius: 6px; font-family: Plus Jakarta Sans; font-size: 11px; font-weight: 600; color: #fff;">${escapeHtml(node.id)} <span style="opacity:.6; font-weight:500;">· ${(TYPE_STYLES[node.type as NodeType] || TYPE_STYLES.concept).label}</span></div>`
        )
        .linkLabel(
          (link: { predicate?: string }) =>
            `<div style="background: rgba(20,20,22,0.9); padding: 4px 8px; border-radius: 6px; font-family: IBM Plex Mono; font-size: 10px; color: #14B8A6;">${escapeHtml(link.predicate || '')}</div>`
        )
        .nodeOpacity(0.95)
        .linkWidth(1.2)
        .linkDirectionalParticles(2)
        .linkDirectionalParticleWidth(1.8)
        .linkDirectionalParticleSpeed(0.006)
        .linkDirectionalArrowLength(4)
        .linkDirectionalArrowRelPos(0.95)
        .onNodeClick((node: { id?: string }) => {
          if (node?.id) onNodeSelectRef.current?.(String(node.id));
        });

      graphRef.current = graph;

      // Defer initial size until layout is ready (host may still be invisible)
      requestAnimationFrame(() => {
        syncSize();
        requestAnimationFrame(syncSize);
      });

      const handleResize = () => syncSize();
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        try {
          graphRef.current?._destructor?.();
        } catch {
          /* ignore */
        }
        graphRef.current = null;
        if (mountRef.current && mountRef.current.parentNode === host) {
          host.removeChild(mountRef.current);
        }
        mountRef.current = null;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Update data / theme when props change
    useEffect(() => {
      const g = graphRef.current;
      if (!g) return;

      const bg = darkMode ? '#0E1015' : '#F4F4F2';
      const node3DColor = (node: { type?: string }) => typeColor(node.type || 'concept', darkMode);

      const existingNodes: Record<string, { id: string; type?: string; degree?: number }> = {};
      try {
        g.graphData()?.nodes?.forEach((n: { id: string }) => {
          existingNodes[n.id] = n;
        });
      } catch {
        /* empty */
      }

      const nodes = rawNodes.map((name) => {
        const n = existingNodes[name] || { id: name };
        n.type = nodeTypes[name] || 'concept';
        n.degree = degrees?.[name] || 0;
        return n;
      });

      const links = (edges || [])
        .map((e) => ({
          source: edgeSource(e),
          target: edgeTarget(e),
          predicate: e.predicate || e.label || 'relates',
        }))
        .filter((l) => rawNodes.includes(l.source) && rawNodes.includes(l.target));

      g.backgroundColor(bg);
      g.nodeColor(node3DColor);
      g.nodeVal((node: { degree?: number }) => 2 + Math.min(node.degree || 0, 10));
      g.linkColor(() =>
        darkMode ? 'rgba(148, 163, 184, 0.45)' : 'rgba(100, 116, 139, 0.45)'
      );
      g.graphData({ nodes, links });
    }, [rawNodes, edges, degrees, nodeTypes, darkMode]);

    // When switching to 3D, restore size + controls (never display:none)
    useEffect(() => {
      if (!visible) return;
      requestAnimationFrame(() => {
        syncSize();
        requestAnimationFrame(syncSize);
      });
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [visible]);

    return (
      <div
        ref={hostRef}
        className="graph-canvas"
        style={{
          // Keep layout size even when hidden so WebGL/controls stay valid
          visibility: visible ? 'visible' : 'hidden',
          pointerEvents: visible ? 'auto' : 'none',
          zIndex: visible ? 1 : 0,
        }}
      />
    );
  }
);

function escapeHtml(s: string) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] || c
  );
}
