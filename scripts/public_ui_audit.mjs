#!/usr/bin/env node

import { readFileSync, existsSync } from 'node:fs'

const checks = [
  {
    name: 'homepage:index',
    file: '/home/zeyufu/Desktop/.work/PeterPonyu-homepage/index.html',
    required: [
      'content="index,follow"',
      'rel="canonical" href="https://peterponyu.github.io/"',
      'property="og:site_name" content="Zeyu Fu"',
      'name="twitter:card" content="summary_large_image"',
      '>Portal<',
      'Open Portal',
    ],
  },
  {
    name: 'homepage:robots',
    file: '/home/zeyufu/Desktop/.work/PeterPonyu-homepage/robots.txt',
    required: [
      'User-agent: *',
      'Allow: /',
      'Sitemap: https://peterponyu.github.io/sitemap.xml',
    ],
  },
  {
    name: 'homepage:sitemap',
    file: '/home/zeyufu/Desktop/.work/PeterPonyu-homepage/sitemap.xml',
    required: [
      'https://peterponyu.github.io/',
      'https://peterponyu.github.io/scportal/',
      'https://peterponyu.github.io/liora-ui/',
      'https://peterponyu.github.io/mrnapp-intersection/',
    ],
  },
  {
    name: 'scportal:app',
    file: '/home/zeyufu/Desktop/.work/peterponyu-readme-audit/scportal/app.vue',
    required: [
      'title ? `${title} | SCPortal`',
      "SCPortal | Single-Cell Discovery Hub",
      "rel: 'canonical'",
      "property: 'og:site_name'",
      "name: 'twitter:card'",
    ],
  },
  {
    name: 'scportal:config',
    file: '/home/zeyufu/Desktop/.work/peterponyu-readme-audit/scportal/nuxt.config.ts',
    required: [
      "title: 'SCPortal | Single-Cell Discovery Hub'",
      "content: 'index,follow'",
      "rel: 'canonical', href: 'https://peterponyu.github.io/scportal/'",
      'https://peterponyu.github.io/assets/badges/scportal.svg',
    ],
  },
  {
    name: 'scportal:navigation',
    file: '/home/zeyufu/Desktop/.work/peterponyu-readme-audit/scportal/components/AppHeader.vue',
    required: [
      "label: 'Homepage'",
      "label: 'LAIOR Benchmarks'",
      "label: 'Utilities'",
    ],
  },
  {
    name: 'liora-ui:layout',
    file: '/home/zeyufu/Desktop/.work/readme-round2-clean/liora-ui/src/app/layout.tsx',
    required: [
      'metadataBase: new URL("https://peterponyu.github.io")',
      'default: "LAIOR Benchmarks | Public Microsite"',
      'canonical: "/liora-ui/"',
      'index: true',
      'follow: true',
    ],
  },
  {
    name: 'liora-ui:home',
    file: '/home/zeyufu/Desktop/.work/readme-round2-clean/liora-ui/src/app/page.tsx',
    required: [
      'Open SCPortal',
      'Back to Homepage',
      'Use this microsite for deep benchmark inspection, and use SCPortal for cross-project discovery.',
    ],
  },
  {
    name: 'mrnapp:index',
    file: '/home/zeyufu/Desktop/.work/peterponyu-readme-audit/mrnapp-intersection/index.html',
    required: [
      '<title>mRNA Intersection | Analysis Utility</title>',
      'rel="canonical" href="https://peterponyu.github.io/mrnapp-intersection/"',
      'Portal',
      'Homepage',
      'twitter:title',
    ],
  },
  {
    name: 'mrnapp:differential',
    file: '/home/zeyufu/Desktop/.work/peterponyu-readme-audit/mrnapp-intersection/differential/index.html',
    required: [
      '<title>mRNA Intersection | Differential Expression</title>',
      'rel="canonical" href="https://peterponyu.github.io/mrnapp-intersection/differential/"',
    ],
  },
  {
    name: 'iAODE:layout',
    file: '/home/zeyufu/Desktop/.work/ui-phase23/iAODE/frontend/src/app/layout.tsx',
    required: [
      "title: 'iAODE Local Training Workspace'",
      'index: false',
      'follow: false',
    ],
  },
  {
    name: 'iAODE:workspace',
    file: '/home/zeyufu/Desktop/.work/ui-phase23/iAODE/frontend/src/app/page.tsx',
    required: [
      'Local-First Training Workspace',
      'Run Locally',
      'Public Pages',
      'SCPortal',
      'localhost:8000',
    ],
  },
  {
    name: 'MCCVAE:landing',
    file: '/home/zeyufu/Desktop/.work/ui-phase23/MCCVAE/out/index.html',
    required: [
      '<title>MCCVAE Local Demo Surface</title>',
      'content="noindex, nofollow"',
      'Local-First Demo Candidate',
      'Back to SCPortal',
      'Back to Homepage',
    ],
  },
  {
    name: 'MCCVAE:404',
    file: '/home/zeyufu/Desktop/.work/ui-phase23/MCCVAE/out/404.html',
    required: [
      '<title>MCCVAE Page Not Found</title>',
      'SCPortal',
      'Homepage',
    ],
  },
]

let failed = false

for (const check of checks) {
  if (!existsSync(check.file)) {
    console.error(`FAIL ${check.name}: missing file ${check.file}`)
    failed = true
    continue
  }

  const text = readFileSync(check.file, 'utf8')
  let localFailed = false
  for (const token of check.required) {
    if (!text.includes(token)) {
      console.error(`FAIL ${check.name}: missing token ${JSON.stringify(token)}`)
      failed = true
      localFailed = true
    }
  }
  if (!localFailed) {
    console.log(`PASS ${check.name}`)
  } else {
    console.log(`CHECKED ${check.name}`)
  }
}

if (failed) {
  process.exit(1)
}
