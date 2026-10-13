// 由矢量母版光栅化 README 品牌资产。
// 用法：node tools/assets/build-assets.mjs
//
// 母版里的标志写成 xlink:href="logo.png" 便于阅读，resvg 不解析相对图片路径，
// 渲染前由本脚本把该引用替换成同目录 logo.png 的 base64 data URI。
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Resvg } from '@resvg/resvg-js';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');

/** 用 resvg 渲染矢量母版到指定位图路径，内联其中的 logo.png 引用。 */
async function renderTo(svgRel, outRel, width) {
  const dir = path.join(repo, path.dirname(svgRel));
  const logo = (await readFile(path.join(dir, 'logo.png'))).toString('base64');
  const svg = (await readFile(path.resolve(repo, svgRel), 'utf8')).replace(
    'href="logo.png"',
    `href="data:image/png;base64,${logo}"`,
  );
  const png = new Resvg(svg, {
    fitTo: { mode: 'width', value: width },
    background: 'rgba(0,0,0,0)',
    font: { loadSystemFonts: true },
  })
    .render()
    .asPng();
  const out = path.resolve(repo, outRel);
  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, png);
  console.log(`${outRel} ← ${svgRel}（宽 ${width}）`);
}

await renderTo('tools/assets/banner.svg', 'assets/banner.png', 1536);
