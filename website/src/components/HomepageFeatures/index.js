import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '设计周期轨道',
    description: (
      <>
        DRO、Halo、Lyapunov — 从初始猜测到收敛轨道，再到整族轨道的延拓。
        微分修正、自然延拓、伪弧长延拓一站式完成。
      </>
    ),
  },
  {
    title: '分析轨道稳定性',
    description: (
      <>
        Floquet 乘子、分岔检测、稳定性指数 — 深入理解轨道的动力学特性。
      </>
    ),
  },
  {
    title: '可视化轨道族',
    description: (
      <>
        2D/3D 投影、Jacobi 着色、稳定性图 — 生成高质量的轨道力学可视化图表。
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
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
