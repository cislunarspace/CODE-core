import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.heroContent}>
          <Heading as="h1" className={styles.heroTitle}>
            {siteConfig.title}
          </Heading>
          <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
          <div className={styles.heroCta}>
            <Link className={styles.ctaPrimary} to="/docs">
              快速上手
            </Link>
            <Link className={styles.ctaSecondary} to="/docs">
              阅读文档
            </Link>
          </div>
        </div>
        <div className={styles.heroCode}>
          <pre className={styles.codeBlock}><code>
{"# 安装\npip install -e .\n\n# 创建地月系统\nfrom e2m2e.core import CR3BP_System\nsystem = CR3BP_System.from_known_system(\"earth_moon\")"}
          </code></pre>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} — 地月转移轨道设计库`}
      description="基于圆型限制性三体问题 (CR3BP) 的轨道力学工具，用于设计地月空间的周期轨道和转移轨道。">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
