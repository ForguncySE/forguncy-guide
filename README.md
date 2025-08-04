# 格言格语

售前团队的文本资料汇集地，活字格技术方案的图书馆。

本工程基于 [vitepress](https://vitepress.dev/guide/what-is-vitepress) 构建。

## 环境依赖

-   node 18+
-   npm
-   Vue 3 + TypeScript + Vite

## 运行

1. 当前目录下，终端运行 `npm install`，安装项目所有依赖。
2. 依赖安装成功后，终端运行 `npm run docs:dev`，启动开发环境。

    > 终端运行 `npm run docs-expose:dev` 可将本地的服务对外暴露。

3. 浏览器访问对应路径即可本地查看文档。

## 结构说明

```text
├── docs
│   ├── .vitepress
│   │   ├── config.mts
│   │   └── meta.mts
│   ├── guide
│   ├── index.md
│   ├── public
│   ├── solution
│   └── standard
└── package.json
```

-   `package.json`: 工程配置
-   `docs`: 文档目录

    -   `.vitepress`: 文档配置目录，其中 `config.ts` 为配置文件，导航、菜单、样式、交互等都在此进行配置。`meta.ts` 为配置所依赖的元数据。
    -   `public`: 静态资源。
    -   `guide`: 学习指南根目录。
    -   `ai`: AI相关根目录。
    -   `solution`: 解决方案根目录。
    -   `standard`: 标准化根目录。
    -   `index.md`: 首页。

## 文档编写说明

1. 文档编写请遵循标准的 `Markdown` 语法。
2. 为确保在 Github 上的文档风格一致，请使用 [Github 风格的警报](https://vitepress.dev/zh/guide/markdown#github-flavored-alerts)
3. 静态资源的引用，请优先选择相对路径。
4. 代码块组 `code-group` 支持[图标标识](https://github.com/yuyinws/vitepress-plugin-group-icons)，可通过文件类型/关键字自动进行识别。如需自定义图标，可在 `config.ts` 中进行配置。

## PDF 导出

文档使用 [`vitepress-export-pdf`](https://github.com/condorheroblog/vitepress-export-pdf) 实现 pdf 导出。

导出配置文件为：`vitepress-pdf.config.ts`。导出时请自行修改 `targetExportPath` 以及 `routePatterns`。

> [!TIP]
> routePatterns 的配置规则是，先配置全部数据，然后排除不需要导出的模块。

导出命令为：

```bash
npm run docs:export
```
