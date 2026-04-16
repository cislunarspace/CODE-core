// @ts-check

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'E2M2E',
  tagline: '地月转移轨道设计库 — Cislunar Transfer Orbit Design Library',
  favicon: 'img/favicon.svg',

  future: {
    v4: true,
  },

  url: 'https://cislunarspace.github.io',
  baseUrl: '/e2m2e/',

  organizationName: 'cislunarspace',
  projectName: 'e2m2e',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans', 'en'],
    localeConfigs: {
      'zh-Hans': {
        label: '简体中文',
        direction: 'ltr',
      },
      en: {
        label: 'English',
        direction: 'ltr',
      },
    },
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/cislunarspace/e2m2e/tree/master/website/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'mbse',
        path: 'mbse',
        routeBasePath: 'mbse',
        sidebarPath: './sidebarsMbse.js',
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
      },
    ],
  ],

  markdown: {
    mermaid: true,
    format: 'md',
    mdx1Compat: {
      comments: false,
      admonitions: false,
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-nB0miv6/jRmo5YADaGePfKy7aP2J7jxLtIPeO7xe+iM0ZHFfDwZDiK0Olk3+fbXpn',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/logo.svg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'E2M2E',
        logo: {
          alt: 'E2M2E Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'mainSidebar',
            position: 'left',
            label: '文档',
          },
          {
            to: '/mbse',
            label: 'MBSE 模型',
            position: 'left',
          },
          {
            href: 'https://github.com/cislunarspace/e2m2e',
            label: 'GitHub',
            position: 'right',
          },
          {
            type: 'localeDropdown',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: '文档',
            items: [
              {
                label: '快速上手',
                to: '/docs',
              },
              {
                label: '微分修正',
                to: '/docs/algorithms/differential_correction',
              },
            ],
          },
          {
            title: '更多',
            items: [
              {
                label: 'MBSE 模型',
                to: '/mbse',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/cislunarspace/e2m2e',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} E2M2E Team. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.oneDark,
        additionalLanguages: ['python'],
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },

    }),
};

export default config;
