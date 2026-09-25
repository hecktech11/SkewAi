import { useId, useMemo, useState } from "react";
import { fmtCount } from "./format.js";

/**
 * SVG chart primitives — no charting dependency.
 *
 * Every chart takes colours from CSS custom properties so it follows the active
 * theme, and exposes `role="img"` with a summary label plus an optional visually
 * hidden data table for screen readers.
 */

function toPoints(data) {
  return (data || []).map((d, i) =>
    typeof d === "number"
      ? { label: String(i), value: d }
      : { label: d.label ?? String(i), value: Number(d.value) || 0, tone: d.tone },
  );
}

function extent(points, { baseZero = true } = {}) {
  if (points.length === 0) return [0, 1];
  const values = points.map((p) => p.value);
  const max = Math.max(...values);
  const min = baseZero ? Math.min(0, ...values) : Math.min(...values);
  if (max === min) return [min, min + 1];
  return [min, max];
}

function linePath(points, w, h, pad, min, max) {
  const span = max - min || 1;
  const stepX = points.length > 1 ? (w - pad * 2) / (points.length - 1) : 0;
  return points
    .map((p, i) => {
      const x = pad + i * stepX;
      const y = h - pad - ((p.value - min) / span) * (h - pad * 2);
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

/** Inline trend line sized to sit inside a table cell or stat card. */
export function Sparkline({ data, width = 120, height = 32, tone = "accent", label }) {
  const points = toPoints(data);
  const [min, max] = extent(points, { baseZero: false });
  if (points.length < 2) {
    return <span className="spark-empty" aria-hidden="true" />;
  }
  const d = linePath(points, width, height, 3, min, max);
  const last = points[points.length - 1];
  const span = max - min || 1;
  const lastX = width - 3;
  const lastY = height - 3 - ((last.value - min) / span) * (height - 6);

  return (
    <svg
      className={`spark tone-${tone}`}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label || `Trend, latest value ${last.value}`}
      preserveAspectRatio="none"
    >
      <path d={d} className="spark-line" fill="none" />
      <circle cx={lastX} cy={lastY} r="2.5" className="spark-dot" />
    </svg>
  );
}

/**
 * Area/line chart with hover readout. Values are read on the nearest x-step so
 * pointer precision does not matter on a wallboard viewed from across a room.
 */
export function AreaChart({
  data,
  height = 180,
  tone = "accent",
  label,
  valueFormat = fmtCount,
  showAxis = true,
}) {
  const gradId = useId();
  const points = useMemo(() => toPoints(data), [data]);
  const [hover, setHover] = useState(null);
  const width = 600;
  const pad = 10;
  const padBottom = showAxis ? 22 : pad;
  const [min, max] = extent(points);

  if (points.length === 0) {
    return <div className="chart-empty">No data in range</div>;
  }

  const span = max - min || 1;
  const stepX = points.length > 1 ? (width - pad * 2) / (points.length - 1) : 0;
  const plotH = height - pad - padBottom;
  const yFor = (v) => height - padBottom - ((v - min) / span) * plotH;
  const line = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${(pad + i * stepX).toFixed(2)},${yFor(p.value).toFixed(2)}`)
    .join(" ");
  const area = `${line} L${(pad + (points.length - 1) * stepX).toFixed(2)},${height - padBottom} L${pad},${
    height - padBottom
  } Z`;

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const rel = ((e.clientX - rect.left) / rect.width) * width;
    const idx = Math.max(0, Math.min(points.length - 1, Math.round((rel - pad) / (stepX || 1))));
    setHover(idx);
  };

  const active = hover === null ? null : points[hover];

  return (
    <div className={`chart tone-${tone}`}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={label || `Trend chart with ${points.length} points, peak ${max}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        preserveAspectRatio="none"
        className="chart-svg"
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" className="chart-grad-top" />
            <stop offset="100%" className="chart-grad-bottom" />
          </linearGradient>
        </defs>

        <line x1={pad} y1={height - padBottom} x2={width - pad} y2={height - padBottom} className="chart-axis" />
        <line x1={pad} y1={yFor(max)} x2={width - pad} y2={yFor(max)} className="chart-grid" />

        <path d={area} fill={`url(#${gradId})`} />
        <path d={line} className="chart-line" fill="none" />

        {active && (
          <g>
            <line
              x1={pad + hover * stepX}
              y1={pad}
              x2={pad + hover * stepX}
              y2={height - padBottom}
              className="chart-cursor"
            />
            <circle cx={pad + hover * stepX} cy={yFor(active.value)} r="3.5" className="chart-dot" />
          </g>
        )}
      </svg>

      <div className="chart-readout" aria-live="polite">
        {active ? (
          <>
            <span className="chart-readout-value mono">{valueFormat(active.value)}</span>
            <span className="chart-readout-label">{active.label}</span>
          </>
        ) : (
          <>
            <span className="chart-readout-value mono">{valueFormat(points[points.length - 1].value)}</span>
            <span className="chart-readout-label">latest · peak {valueFormat(max)}</span>
          </>
        )}
      </div>
    </div>
  );
}

/** Horizontal ranked bars — the right shape for "top clusters" style lists. */
export function BarList({ data, valueFormat = fmtCount, tone = "accent", max: maxOverride, emptyText = "No data" }) {
  const points = toPoints(data);
  if (points.length === 0) return <div className="chart-empty">{emptyText}</div>;
  const max = maxOverride ?? Math.max(...points.map((p) => p.value), 1);

  return (
    <ul className={`barlist tone-${tone}`}>
      {points.map((p, i) => (
        <li key={`${p.label}-${i}`} className="barlist-row">
          <div className="barlist-head">
            <span className="barlist-label" title={p.label}>
              {p.label}
            </span>
            <span className="barlist-value mono">{valueFormat(p.value)}</span>
          </div>
          <div className="barlist-track">
            <div
              className={`barlist-fill${p.tone ? ` tone-${p.tone}` : ""}`}
              style={{ width: `${Math.max(2, (p.value / max) * 100)}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Segmented distribution bar — severity or funnel mix in one line. */
export function StackBar({ segments, label }) {
  const items = (segments || []).filter((s) => Number(s.value) > 0);
  const total = items.reduce((sum, s) => sum + Number(s.value), 0);
  if (total === 0) return <div className="chart-empty">No distribution yet</div>;

  return (
    <div className="stackbar-wrap">
      <div className="stackbar" role="img" aria-label={label || "Distribution"}>
        {items.map((s) => (
          <div
            key={s.label}
            className={`stackbar-seg tone-${s.tone || "accent"}`}
            style={{ width: `${(s.value / total) * 100}%` }}
            title={`${s.label}: ${s.value}`}
          />
        ))}
      </div>
      <ul className="stackbar-legend">
        {items.map((s) => (
          <li key={s.label}>
            <span className={`legend-dot tone-${s.tone || "accent"}`} aria-hidden="true" />
            {s.label}
            <span className="mono legend-value">{s.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Radial gauge for a single 0–1 ratio (health, coverage, hit rate). */
export function Gauge({ value, label, tone = "accent", size = 104 }) {
  const ratio = Math.max(0, Math.min(1, Number(value) || 0));
  const stroke = 8;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const dash = circumference * ratio;

  return (
    <div className={`gauge tone-${tone}`} style={{ width: size }}>
      <svg width={size} height={size} role="img" aria-label={`${label}: ${Math.round(ratio * 100)}%`}>
        <circle cx={size / 2} cy={size / 2} r={r} className="gauge-track" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          className="gauge-fill"
          strokeWidth={stroke}
          fill="none"
          strokeDasharray={`${dash} ${circumference - dash}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="gauge-center">
        <span className="gauge-value mono">{Math.round(ratio * 100)}%</span>
      </div>
      <div className="gauge-label">{label}</div>
    </div>
  );
}

function toneStroke(tone) {
  if (tone === "danger") return "var(--danger)";
  if (tone === "warn") return "var(--warn)";
  if (tone === "ok") return "var(--ok)";
  return "var(--accent)";
}

/** Donut chart for share-of-total breakdowns (verdicts, channels). */
export function DonutChart({ data, size = 132, thickness = 20, label }) {
  const items = (data || [])
    .map((d) => ({ label: d.label ?? "", value: Number(d.value) || 0, tone: d.tone }))
    .filter((d) => d.value > 0);
  const total = items.reduce((sum, d) => sum + d.value, 0);
  if (items.length === 0) return <div className="chart-empty">No data</div>;
  const r = (size - thickness) / 2;
  const c = size / 2;
  let acc = 0;
  return (
    <div className="donut-flex">
      <svg
        width={size}
        height={size}
        role="img"
        aria-label={label || `Donut chart, ${items.length} segments, total ${total}`}
      >
        <circle cx={c} cy={c} r={r} fill="none" stroke="var(--bg-hover)" strokeWidth={thickness} />
        {items.map((d, i) => {
          const frac = d.value / total;
          const el = (
            <circle
              key={`${d.label}-${i}`}
              cx={c}
              cy={c}
              r={r}
              fill="none"
              stroke={toneStroke(d.tone)}
              strokeWidth={thickness}
              pathLength={100}
              strokeDasharray={`${(frac * 100).toFixed(2)} ${(100 - frac * 100).toFixed(2)}`}
              strokeDashoffset={(-acc * 100).toFixed(2)}
              transform={`rotate(-90 ${c} ${c})`}
            >
              <title>{`${d.label}: ${d.value}`}</title>
            </circle>
          );
          acc += frac;
          return el;
        })}
      </svg>
      <ul className="stackbar-legend">
        {items.map((d, i) => (
          <li key={`${d.label}-${i}`}>
            <span className={`legend-dot tone-${d.tone || "accent"}`} aria-hidden="true" />
            {d.label}
            <span className="mono legend-value">
              {d.value} · {Math.round((d.value / total) * 100)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Two-dimensional intensity matrix (e.g. severity × status). */
export function HeatMatrix({ rows, columns, getValue, formatValue = (v) => v, label }) {
  const rs = rows || [];
  const cs = columns || [];
  if (rs.length === 0 || cs.length === 0) return <div className="chart-empty">No data</div>;
  const max = Math.max(
    1,
    ...rs.flatMap((r) => cs.map((c) => Number(getValue(r.key, c.key)) || 0)),
  );
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="heat-table">
        <caption className="sr-only">{label || "Matrix"}</caption>
        <thead>
          <tr>
            <th scope="col">
              <span className="sr-only">Row</span>
            </th>
            {cs.map((c) => (
              <th key={c.key} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rs.map((r) => (
            <tr key={r.key}>
              <th scope="row">{r.label}</th>
              {cs.map((c) => {
                const v = Number(getValue(r.key, c.key)) || 0;
                const intensity = Math.round((v / max) * 80);
                return (
                  <td
                    key={c.key}
                    style={{
                      background:
                        v > 0
                          ? `color-mix(in srgb, var(--accent) ${intensity}%, transparent)`
                          : undefined,
                    }}
                    title={`${r.label} × ${c.label}: ${v}`}
                  >
                    {v > 0 ? formatValue(v) : <span className="heat-zero">·</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
