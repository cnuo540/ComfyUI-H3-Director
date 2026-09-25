/* H3 导演台 · 每段生成上限（档位）唯一数据源。
   档位由 H3DirectorStudio 节点的「每段生成上限」下拉选择并随工作流保存；
   h3_studio.js / h3_script_tools.js / h3_guided_templates.js 共享本状态。
   frames 必须是 H3 的 17k+5 合法帧网格点：7→175（17×10+5）、10→243（17×14+5）、15→362（17×21+5）。
   maxgen = frames / 24，是导入分桶用的秒上限（15 档即原版 15.083s；7 档 = 175/24 ≈ 7.29s）。 */

export const H3_CAP_TIERS = Object.freeze({
  "7秒": Object.freeze({
    label: "7秒", sec: 7, frames: 175, maxgen: 175 / 24, clamp: 7, refFrames: 24 * 7,
    split: Object.freeze({ min: 5, max: 7, target: 6 }),
  }),
  "10秒": Object.freeze({
    label: "10秒", sec: 10, frames: 243, maxgen: 243 / 24, clamp: 10, refFrames: 24 * 10,
    split: Object.freeze({ min: 6, max: 10, target: 8 }),
  }),
  "15秒": Object.freeze({
    label: "15秒", sec: 15, frames: 362, maxgen: 362 / 24, clamp: 15, refFrames: 24 * 15,
    split: Object.freeze({ min: 8, max: 15, target: 12 }),
  }),
});

/* 可变共享状态；默认 7 秒，与节点 combo 第一项（默认值）一致。
   旧工作流传的 "8秒" 不在表内 → setH3Cap 兜底落到 7 秒档（与后端 _set_duration_cap 一致）。 */
export const H3_CAP = Object.assign({}, H3_CAP_TIERS["7秒"]);

export function setH3Cap(label) {
  const raw = String(label == null ? "" : label).trim();
  const key = H3_CAP_TIERS[raw] ? raw
    : (/^\d+$/.test(raw) && H3_CAP_TIERS[raw + "秒"] ? raw + "秒" : "");
  Object.assign(H3_CAP, H3_CAP_TIERS[key] || H3_CAP_TIERS["7秒"]);
  return H3_CAP;
}
