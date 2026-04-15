/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  mainSidebar: [
    'intro',
    {
      type: 'category',
      label: '快速上手',
      items: [
        'guides/orbit-generation',
        'algorithms/stability',
        'guides/visualization-guide',
      ],
    },
    {
      type: 'category',
      label: '深入了解',
      items: [
        'core/system',
        'core/dynamics',
        'core/orbit',
        'algorithms/differential_correction',
        'algorithms/continuation',
        'algorithms/halo',
      ],
    },
    {
      type: 'category',
      label: '技术参考',
      items: [
        'reference/api-reference',
        'reference/algorithms',
      ],
    },
    {
      type: 'category',
      label: '其他',
      items: [
        'core/coordinate',
        'core/ephemeris_system',
        'core/ephemeris_dynamics',
        'core/spice',
        'algorithms/multiple_shooting',
        'guides/system-overview',
        'visualization/plotting',
        'guides/release',
      ],
    },
  ],
};

export default sidebars;
