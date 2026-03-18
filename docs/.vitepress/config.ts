import { DefaultTheme, defineConfig } from 'vitepress'
import { groupIconMdPlugin, groupIconVitePlugin, localIconLoader } from 'vitepress-plugin-group-icons'
import { baseUrl, siteDescription, siteTitle } from './meta'

/* 顶部的导航菜单 */
const nav: DefaultTheme.Config['nav'] = [
  {
    text: '标准化',
    link: '/standard/introduction',
    activeMatch: '^/standard'
  },
  {
    text: `关于`,
    items: [
      {
        text: '活字格官网',
        link: 'https://www.grapecity.com.cn/solutions/huozige'
      },
      {
        text: '开发者学堂',
        link: 'https://learn.grapecity.com.cn/home'
      },
      {
        text: '葡萄城市场',
        link: 'https://marketplace.grapecity.com.cn/'
      }
    ]
  },
]

/* 左侧菜单 */
const sidebar: DefaultTheme.Config['sidebar'] = {
  '/standard/': siderbarStandard(),
}

// 标准化的左侧菜单
function siderbarStandard() {
  return [
    { text: '简介', link: '/standard/introduction' },
    { text: '术语说明', link: '/standard/idioms' },
    {
      text: '架构',
      items: [
        { text: '系统架构', link: '/standard/arch/sys-arch' },
        { text: '生命周期', link: '/standard/arch/life-cycle' },
        { text: '部署方案', link: '/standard/arch/deployment' },
        { text: '可用性', link: '/standard/arch/robustness' },
      ]
    },
    {
      text: 'UI & UX',
      items: [
        { text: '设计原则', link: '/standard/design/principle' },
        {
          text: '样式',
          collapsed: true,
          items: [
            { text: '主题色', link: '/standard/design/style/theme' },
            { text: '布局', link: '/standard/design/style/layout' },
            { text: '字体', link: '/standard/design/style/typeface' },
          ]
        },
        {
          text: '模式',
          collapsed: true,
          items: [
            { text: '距离', link: '/standard/design/model/distance' },
            { text: '对齐', link: '/standard/design/model/alignment' },
            { text: '页面', link: '/standard/design/model/page' },
            { text: '反馈', link: '/standard/design/model/feedback' },
          ]
        },
      ]
    },
    {
      text: '开发',
      items: [
        { text: '开发原则', link: '/standard/dev/principle' },
        {
          text: '基础规约',
          collapsed: true,
          items: [
            { text: '命名风格', link: '/standard/dev/base/naming-style' },
            { text: '界面设计', link: '/standard/dev/base/page-design' },
            { text: '业务处理', link: '/standard/dev/base/business' },
            { text: '异常捕获', link: '/standard/dev/base/exception' },
            { text: '组件复用', link: '/standard/dev/base/component' },
            { text: '文档注释', link: '/standard/dev/base/docs' },
          ],
        },
        {
          text: '数据库',
          collapsed: true,
          items: [
            { text: '类型选择', link: '/standard/dev/database/selection' },
            { text: '模型设计', link: '/standard/dev/database/domain' },
            { text: '数据检索', link: '/standard/dev/database/retrieval' }
          ],
        },
        { text: '工程扩展', link: '/standard/dev/expansion' },
        { text: '协同开发', link: '/standard/dev/cooperation' },
        {
          text: '配置管理',
          collapsed: true,
          items: [
            { text: '用户权限', link: '/standard/dev/configuration/permission' },
            { text: '应用配置', link: '/standard/dev/configuration/application' },
          ],
        },

      ]
    },
    {
      text: '运维',
      items: [
        { text: '系统维护', link: '/standard/operation/system' },
        { text: '数据治理', link: '/standard/operation/data' },
        { text: '服务配置', link: '/standard/operation/service' },
      ],
    },
  ]
}


/* 网站配置 */
export default defineConfig({

  base: baseUrl,
  title: siteTitle,
  description: siteDescription,

  head: [
    ['link', { rel: 'icon', href: '/forguncy-icon.png' }],
    [
      'script',
      { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-JPQJY2Z57W' }
    ],
    [
      'script',
      {},
      `window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-JPQJY2Z57W');`
    ]
  ],

  lastUpdated: true,
  cleanUrls: true,

  /* markdown 配置 */
  markdown: {
    container: {
      tipLabel: 'TIP',
      warningLabel: 'WANNING',
      dangerLabel: 'DANGER',
      infoLabel: 'NOTE',
      detailsLabel: 'DETAILS'
    },
    lineNumbers: true,
    config(md) {
      md.use(groupIconMdPlugin)
    }
  },
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN'
    },
  },

  /* 主题配置 */
  themeConfig: {
    i18nRouting: false,
    logo: '/forguncy-icon.png',
    search: {
      provider: 'local',
    },

    /* 当前页右侧导航目录 */
    outline: {
      level: 'deep',
      label: '当前页导航'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '外观',
    sidebarMenuLabel: '目录',

    nav,

    sidebar,

    // editLink: {
    //   pattern: `${github}/tree/master/docs/:path`,
    //   text: '在 GitHub 上编辑此页'
    // },

    lastUpdated: {
      text: '最后更新'
    },

    // socialLinks: [
    //   {icon: 'github', link: 'https://github.com/vuejs/vitepress'}
    // ],

    /*备案号*/
    // <span style="display:inline-block;;height: 16px;width: 16px;fill: #5791ED;position: relative;top: 3px;"><svg viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg"><title>ICP备案号</title><path d="M778.24 163.84c-76.8-40.96-165.888-61.44-269.312-61.44s-192.512 20.48-269.312 61.44h-133.12l23.552 337.92c8.192 113.664 67.584 217.088 162.816 280.576l215.04 144.384 215.04-144.384c96.256-63.488 155.648-166.912 163.84-280.576l23.552-337.92H778.24z m47.104 333.824c-7.168 94.208-56.32 181.248-135.168 233.472l-181.248 120.832L327.68 731.136c-78.848-53.248-129.024-139.264-135.168-233.472L173.056 225.28h136.192v-26.624c58.368-23.552 124.928-34.816 199.68-34.816s141.312 12.288 199.68 34.816V225.28H844.8l-19.456 272.384z"></path><path d="M685.056 328.704v-46.08H455.68c2.048-4.096 6.144-9.216 11.264-15.36 5.12-7.168 9.216-12.288 11.264-15.36L419.84 240.64c-31.744 46.08-75.776 87.04-133.12 123.904 4.096 4.096 10.24 11.264 18.432 21.504l17.408 17.408c23.552-15.36 45.056-31.744 63.488-50.176 26.624 25.6 49.152 43.008 67.584 51.2-46.08 15.36-104.448 27.648-175.104 35.84 2.048 5.12 6.144 13.312 9.216 24.576 4.096 11.264 6.144 19.456 7.168 24.576l39.936-7.168v218.112H389.12V680.96h238.592v19.456h54.272V481.28H348.16c60.416-12.288 114.688-27.648 163.84-46.08 49.152 19.456 118.784 34.816 210.944 46.08 5.12-17.408 10.24-34.816 17.408-51.2-62.464-4.096-116.736-12.288-161.792-24.576 38.912-20.48 74.752-46.08 106.496-76.8z m-150.528 194.56h94.208v41.984h-94.208v-41.984z m0 78.848h94.208v41.984h-94.208v-41.984z m-144.384-78.848h94.208v41.984H390.144v-41.984z m0 78.848h94.208v41.984H390.144v-41.984zM424.96 326.656h182.272c-26.624 22.528-57.344 41.984-94.208 57.344-31.744-15.36-61.44-34.816-88.064-57.344z"></path></svg></span>' +
    //     '<span>备案号：<a href="https://beian.miit.gov.cn">陕ICP备2023004798号-2</a></span><br/>

    footer: {
      message: '<span>Released under the MIT License.</span>',
      copyright: 'Copyright © 2023-present 格言格语'
    },

  },

  vite: {
    plugins: [
      groupIconVitePlugin({
        customIcon: {
          'java': 'vscode-icons:file-type-java',
          'pdm': 'vscode-icons:file-type-pdm',
          'node': 'vscode-icons:file-type-node',
          'python': 'vscode-icons:file-type-python',
        },
      })
    ]
  }
})
