#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const desktopRoot = path.resolve(scriptDir, '..', '..');

const surfaces = [
  {
    name: 'Homepage',
    file: path.join(desktopRoot, '.work/PeterPonyu-homepage/index.html'),
    checks: [
      ['canonical URL', 'rel="canonical" href="https://peterponyu.github.io/"'],
      ['robots index/follow', 'name="robots" content="index,follow"'],
      ['SCPortal naming', '>SCPortal<'],
      ['Open SCPortal CTA', 'Open SCPortal'],
      ['mRNA Intersection label', 'mRNA Intersection'],
    ],
  },
  {
    name: 'SCPortal',
    file: path.join(desktopRoot, '.work/peterponyu-readme-audit/scportal/nuxt.config.ts'),
    checks: [
      ['title contract', "title: 'SCPortal | Single-Cell Discovery Hub'"],
      ['robots index/follow', "name: 'robots', content: 'index,follow'"],
      ['canonical URL', "rel: 'canonical', href: 'https://peterponyu.github.io/scportal/'"],
      ['OG title', "property: 'og:title', content: 'SCPortal | Single-Cell Discovery Hub'"],
      ['Twitter title', "name: 'twitter:title', content: 'SCPortal | Single-Cell Discovery Hub'"],
    ],
  },
  {
    name: 'SCPortal nav copy',
    file: path.join(desktopRoot, '.work/peterponyu-readme-audit/scportal/components/AppHeader.vue'),
    checks: [
      ['Homepage label', "label: 'Homepage'"],
      ['LAIOR Benchmarks label', "label: 'LAIOR Benchmarks'"],
      ['mRNA Intersection label', "label: 'mRNA Intersection'"],
    ],
  },
  {
    name: 'LAIOR Benchmarks',
    file: path.join(desktopRoot, '.work/readme-round2-clean/liora-ui/src/app/layout.tsx'),
    checks: [
      ['title contract', 'default: "LAIOR Benchmarks | Public Microsite"'],
      ['canonical URL', 'canonical: "/liora-ui/"'],
      ['robots index/follow', 'index: true'],
      ['OG site name', 'siteName: "LAIOR Benchmarks"'],
      ['Twitter card', 'card: "summary_large_image"'],
    ],
  },
  {
    name: 'mRNA Intersection root',
    file: path.join(desktopRoot, '.work/peterponyu-readme-audit/mrnapp-intersection/index.html'),
    checks: [
      ['title contract', '<title>mRNA Intersection | Analysis Utility</title>'],
      ['robots index/follow', 'name="robots" content="index,follow"'],
      ['canonical URL', 'rel="canonical" href="https://peterponyu.github.io/mrnapp-intersection/"'],
      ['SCPortal naming', '>SCPortal</a>'],
    ],
  },
  {
    name: 'mRNA Intersection differential',
    file: path.join(desktopRoot, '.work/peterponyu-readme-audit/mrnapp-intersection/differential/index.html'),
    checks: [
      ['robots index/follow', 'name="robots" content="index,follow"'],
      ['SCPortal naming', '>SCPortal</a>'],
    ],
  },
  {
    name: 'mRNA Intersection enrichment',
    file: path.join(desktopRoot, '.work/peterponyu-readme-audit/mrnapp-intersection/enrichment/index.html'),
    checks: [
      ['robots index/follow', 'name="robots" content="index,follow"'],
      ['SCPortal naming', '>SCPortal</a>'],
    ],
  },
  {
    name: 'mRNA Intersection intersection',
    file: path.join(desktopRoot, '.work/peterponyu-readme-audit/mrnapp-intersection/intersection/index.html'),
    checks: [
      ['robots index/follow', 'name="robots" content="index,follow"'],
      ['SCPortal naming', '>SCPortal</a>'],
    ],
  },
  {
    name: 'mRNA Intersection gene search',
    file: path.join(desktopRoot, '.work/peterponyu-readme-audit/mrnapp-intersection/gene-search/index.html'),
    checks: [
      ['robots index/follow', 'name="robots" content="index,follow"'],
      ['SCPortal naming', '>SCPortal</a>'],
    ],
  },
  {
    name: 'iAODE workspace metadata',
    file: path.join(desktopRoot, '.work/ui-phase23/iAODE/frontend/src/app/layout.tsx'),
    checks: [
      ['title contract', "default: 'iAODE Workspace | Local-First Training'"],
      ['canonical URL', "canonical: '/iAODE/frontend/'"],
      ['robots noindex', 'index: false'],
      ['OG title', "title: 'iAODE Workspace | Local-First Training'"],
    ],
  },
  {
    name: 'iAODE workspace copy',
    file: path.join(desktopRoot, '.work/ui-phase23/iAODE/frontend/src/app/page.tsx'),
    checks: [
      ['Homepage backlink', 'href="https://peterponyu.github.io/"'],
      ['Public Pages backlink', 'href="https://peterponyu.github.io/iAODE/"'],
      ['SCPortal backlink', 'href="https://peterponyu.github.io/scportal/"'],
      ['local-first badge', 'Local-First Training Workspace'],
    ],
  },
  {
    name: 'MCCVAE landing',
    file: path.join(desktopRoot, '.work/ui-phase23/MCCVAE/out/index.html'),
    checks: [
      ['robots noindex', '<meta name="robots" content="noindex, nofollow">'],
      ['Homepage backlink', '>Homepage<'],
      ['SCPortal backlink', '>SCPortal<'],
      ['local-first label', 'Local-First Demo Candidate'],
    ],
  },
];

let failed = false;

for (const surface of surfaces) {
  let content;
  try {
    content = fs.readFileSync(surface.file, 'utf8');
  } catch (error) {
    failed = true;
    console.error(`FAIL ${surface.name}: missing file ${surface.file}`);
    continue;
  }

  for (const [label, token] of surface.checks) {
    if (!content.includes(token)) {
      failed = true;
      console.error(`FAIL ${surface.name}: missing ${label}`);
    }
  }
}

if (failed) {
  process.exitCode = 1;
} else {
  console.log('Public UI audit passed.');
}
