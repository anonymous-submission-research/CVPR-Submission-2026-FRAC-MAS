"use client";

import React from "react";

interface Props {
  matrix: number[][]; // rows = true classes, cols = predicted
  labels: string[];
  medicalLight?: boolean;
}

export default function ConfusionMatrix({
  matrix,
  labels,
  medicalLight,
}: Props) {
  const max = Math.max(...matrix.flat(), 1);
  const isLight = !!medicalLight;
  return (
    <div className="overflow-auto">
      <table
        className={`table-auto border-collapse text-sm w-full ${
          isLight ? "border-neutral-200" : "border-neutral-700"
        }`}
      >
        <thead>
          <tr>
            <th
              className={`border p-2 text-left ${
                isLight ? "bg-slate-100 text-slate-800" : "bg-neutral-900/10"
              }`}
            >
              True \ Pred
            </th>
            {labels.map((l) => (
              <th
                key={l}
                className={`border p-2 text-left ${
                  isLight ? "bg-slate-50 text-slate-800" : "bg-neutral-900/5"
                }`}
              >
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td
                className={`border p-2 font-medium ${
                  isLight ? "bg-slate-50 text-slate-800" : "bg-neutral-900/5"
                }`}
              >
                {labels[i]}
              </td>
              {row.map((v, j) => {
                const intensity = Math.min(0.9, v / max || 0);
                const bg = isLight
                  ? `rgba(6,78,59, ${0.06 + intensity * 0.55})`
                  : `rgba(14,116,144, ${0.08 + intensity * 0.75})`;
                const textColor = isLight
                  ? intensity > 0.45
                    ? "#fff"
                    : "#0f172a"
                  : intensity > 0.45
                  ? "#fff"
                  : "#E6EEF3";
                return (
                  <td
                    key={j}
                    className="border p-2 text-center"
                    style={{ backgroundColor: bg, color: textColor }}
                  >
                    {v}
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
