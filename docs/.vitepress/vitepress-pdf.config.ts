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

  if (config.sidebar) {
    const sidebar = config.sidebar as Record<string, DefaultTheme.SidebarItem[]>
    Object.values(sidebar).forEach(sidebarItems => {
      extractLinks(sidebarItems)
    })
  }
  return links
}

const links = extractLinksFromConfig(userConfig.themeConfig!)

const targetExportPath = ['/standard/']

function filterRoutesByPaths(routes: string[], paths: string[]): string[] {
  return routes.filter(route =>
    paths.some(path => route.startsWith(path))
  );
}

const exportPaths = filterRoutesByPaths(links, targetExportPath).map(path => {
  const base = path.replace(/\.html$/, '')
  return base.startsWith('/') ? base : `/${base}`
})

const headerTemplate = `<div style="margin-top: -0.4cm; height: 70%; width: 100%; display: flex; justify-content: center; align-items: center; color: lightgray; border-bottom: solid lightgray 1px; font-size: 10px;">
  <span class="title"></span>
</div>`

const footerTemplate = `<div style="margin-bottom: -0.4cm; height: 70%; width: 100%; display: flex; justify-content: flex-start; align-items: center; color: lightgray; border-top: solid lightgray 1px; font-size: 10px;">
  <span style="margin-left: 15px;" class="url"></span>
</div>`

export default defineUserConfig({
  outFile: '活字格标准化.pdf',
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
  urlOrigin: 'http://localhost:5173',
  sorter: (pageA, pageB) => {
    const aPath = pageA.path.startsWith('/forguncy-guide/') 
      ? pageA.path.slice('/forguncy-guide'.length) 
      : pageA.path
    const bPath = pageB.path.startsWith('/forguncy-guide/') 
      ? pageB.path.slice('/forguncy-guide'.length) 
      : pageB.path
    const aIndex = exportPaths.findIndex(route => route === aPath)
    const bIndex = exportPaths.findIndex(route => route === bPath)
    return aIndex - bIndex
  },
  routePatterns: [
    ...targetExportPath.map(path => `${path}**`),
    '!/404.html'
  ]
});