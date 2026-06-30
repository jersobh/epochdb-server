import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "EpochDB Docs",
  description: "High-performance sharded memory database for autonomous agents.",
  themeConfig: {
    logo: '/logo.png',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/what-is-epochdb' },
      { text: 'API Reference', link: '/api/' }
    ],
    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'What is EpochDB?', link: '/guide/what-is-epochdb' },
          { text: 'Getting Started', link: '/guide/getting-started' }
        ]
      },
      {
        text: 'Architecture',
        items: [
          { text: 'Sharding & Hashing', link: '/guide/sharding-architecture' },
          { text: 'Metrics & Health-Based Routing', link: '/guide/routing-metrics' }
        ]
      },
      {
        text: 'User Interface',
        items: [
          { text: 'Visualization Dashboard', link: '/guide/visualization' }
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Endpoints', link: '/api/' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/jersobh/epochdb-server' }
    ]
  }
})
