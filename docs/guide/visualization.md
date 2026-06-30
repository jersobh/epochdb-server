# Visualization Dashboard

The built-in EpochDB visualization panel provides a premium, interactive interface to manage the database cluster.

## Features

- **3D Knowledge Graph**: Renders extracted entities as interactive spheres and relationship triples as directional arrows. Left-click and drag to rotate, scroll to zoom, and right-click to pan.
- **Node-by-Node Metrics**: Displays CPU, RAM, and Disk utilization for each shard container in real-time.
- **Live Cluster Warning Alerts**: Displays high-visibility notifications at the top of the sidebar if any shard goes offline or runs out of resources.
- **Grouped Bar Charts**: Chart.js graphs compare resource usage across nodes side-by-side.
- **Auto-polling Update Loop**: Stats and knowledge graphs refresh automatically in-place every 5 seconds without resets or camera layout jumps.
- **Deleting Memories**: Left-click on any memory delete cross (`&times;`) to delete it. Toggle the **Hard delete** checkbox in Admin Operations to specify whether to soft-delete (hide) or hard-delete (permanently remove) memory atoms.
