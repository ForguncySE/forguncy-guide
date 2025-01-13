import type {DefaultTheme} from 'vitepress'
import { defineUserConfig } from "vitepress-export-pdf"

import userConfig from './config.js'

function extractLinksFromConfig(config: DefaultTheme.Config) {
  const links: string[] = []

  function extractLinks(sidebar: DefaultTheme.SidebarItem[]) {
    for (const item of sidebar) {
      if (item.items)
        extractLinks(item.items)

      else if (item.link)
        links.push(`${item.link}.html`)
    }
  }

  for (const key in config.sidebar)
    extractLinks(config.sidebar[key])
  return links
}

const links = extractLinksFromConfig(userConfig.themeConfig!)

const targetExportPath = ['/solution/gateway']

function filterRoutesByPaths(routes: string[], paths: string[]): string[] {
  return routes.filter(route =>
    paths.some(path => route.startsWith(path))
  );
}

const exportPaths = filterRoutesByPaths(links, targetExportPath)

const headerTemplate = `<div style="margin-top: -0.4cm; height: 70%; width: 100%; display: flex; justify-content: center; align-items: center; color: lightgray; border-bottom: solid lightgray 1px; font-size: 10px;">
  <span class="title"></span>
</div>`

const footerTemplate = `<div style="margin-bottom: -0.4cm; height: 70%; width: 100%; display: flex; justify-content: flex-start; align-items: center; color: lightgray; border-top: solid lightgray 1px; font-size: 10px;">
  <span style="margin-left: 15px;" class="url"></span>
</div>`

export default defineUserConfig({
  outFile: '格言格语-网关方案.pdf',
  outDir: 'output-pdf',
  pdfOptions: {
    format: 'A4',
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate,
    footerTemplate,
    margin: {
      bottom: 60,
      left: 25,
      right: 25,
      top: 60,
    },
  },
  urlOrigin: 'https://forguncyse.github.io',
  sorter: (pageA, pageB) => {
    const aIndex = exportPaths.findIndex(route => { 
      const path = route.endsWith('.html') ? route.slice(0, -5) : route
      return path === pageA.path
    })
    const bIndex = exportPaths.findIndex(route => { 
      const path = route.endsWith('.html') ? route.slice(0, -5) : route
      return path === pageB.path
    })
    return aIndex - bIndex
  },
  routePatterns: [
    '**',
    '!/forguncy-guide/index',
    '!/forguncy-guide/lb-index',
    '!/forguncy-guide/guide/**',
    '!/forguncy-guide/standard/**',
    '!/forguncy-guide/solution/load-balance/**',
    // '!/forguncy-guide/solution/gateway/**',
    '!/forguncy-guide/solution/log-monitor/**',
    '!/404.html'
  ]
});