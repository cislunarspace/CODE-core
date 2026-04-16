import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '设计周期轨道',
    svg: (
      <svg viewBox="0 0 48 48" width="48" height="48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="24" cy="24" r="6" stroke="currentColor" strokeWidth="1.5"/>
        <ellipse cx="24" cy="24" rx="20" ry="9" stroke="currentColor" strokeWidth="1.5" transform="rotate(-15 24 24)"/>
        <circle cx="40" cy="17" r="3" fill="currentColor" opacity="0.6"/>
      </svg>
    ),
    description: (
      <>
        DRO、Halo、Lyapunov — 从初始猜测到收敛轨道，再到整族轨道的延拓。
        微分修正、自然延拓、伪弧长延拓一站式完成。
      </>
    ),
  },
  {
    title: '分析轨道稳定性',
    svg: (
      <svg viewBox="0 0 48 48" width="48" height="48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M6 36 L14 20 L22 28 L30 12 L42 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <circle cx="14" cy="20" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="30" cy="12" r="2" fill="currentColor" opacity="0.4"/>
      </svg>
    ),
    description: (
      <>
        Floquet 乘子、分岔检测、稳定性指数 — 深入理解轨道的动力学特性。
      </>
    ),
  },
  {
    title: '可视化轨道族',
    svg: (
      <svg viewBox="0 0 48 48" width="48" height="48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="6" y="6" width="36" height="36" rx="4" stroke="currentColor" strokeWidth="1.5"/>
        <circle cx="24" cy="24" r="10" stroke="currentColor" strokeWidth="1" opacity="0.5"/>
        <circle cx="24" cy="24" r="4" fill="currentColor" opacity="0.6"/>
        <line x1="24" y1="6" x2="24" y2="14" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
        <line x1="6" y1="24" x2="14" y2="24" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
      </svg>
    ),
    description: (
      <>
        2D/3D 投影、Jacobi 着色、稳定性图 — 生成高质量的轨道力学可视化图表。
      </>
    ),
  },
];

function Feature({title, svg, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>{svg}</div>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
