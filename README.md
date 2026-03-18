# 活字格标准化

活字格开发与运维的最佳实践基础标准。您可以参考团队实际情况做定制调整。

本工程基于 [vitepress](https://vitepress.dev/guide/what-is-vitepress) 构建。

## 环境依赖

- node 18+
- npm
- Vue 3 + TypeScript + Vite

## 运行

1. 当前目录下，终端运行 `npm install`，安装项目所有依赖。
2. 依赖安装成功后，终端运行 `npm run docs:dev`，启动开发环境。
   > 终端运行 `npm run docs-expose:dev` 可将本地的服务对外暴露。
3. 浏览器访问对应路径即可本地查看文档。

## 结构说明

```text
├── docs
│   ├── .vitepress        # vitepress 配置
│   │   ├── config.mts
│   │   └── meta.mts
│   ├── index.md          # 网站首页
│   ├── public            # 网站公共资源
│   └── standard          # 标准化文档
└── package.json
```

## 文档编写说明

1. 文档编写请遵循标准的 `Markdown` 语法。
2. 为确保在 Github 上的文档风格一致，请使用 [Github 风格的警报](https://vitepress.dev/zh/guide/markdown#github-flavored-alerts)
3. 静态资源的引用，请优先选择相对路径。
4. 代码块组 `code-group` 支持[图标标识](https://github.com/yuyinws/vitepress-plugin-group-icons)，可通过文件类型/关键字自动进行识别。如需自定义图标，可在 `config.ts` 中进行配置。

## PDF 导出

文档使用 [`vitepress-export-pdf`](https://github.com/condorheroblog/vitepress-export-pdf) 实现 pdf 导出。

导出配置文件为：`vitepress-pdf.config.ts`。导出时请自行修改 `targetExportPath` 以及 `routePatterns`。

> \[!TIP]
> routePatterns 的配置规则是，先配置全部数据，然后排除不需要导出的模块。

导出命令为：

```bash
npm run docs:export
```

