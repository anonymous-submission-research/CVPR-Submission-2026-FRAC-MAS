"use client";

import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  BarController,
  LineController,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  BarController,
  LineController,
  Tooltip,
  Legend
);

interface Props {
  bins: number[]; // predicted probability bin mids (0..1)
  predicted: number[]; // predicted frequency (0..1)
  observed: number[]; // observed frequency (0..1)
  medicalLight?: boolean;
}

export default function ReliabilityChart({
  bins,
  predicted,
  observed,
  medicalLight,
}: Props) {
  const isLight = !!medicalLight;

  const labels = bins.map((b) => `${Math.round(b * 100)}%`);

  // Keep data in 0..1 range so Brier score (0..1) aligns with chart values
  const data = {
    labels,
    datasets: [
      {
        type: "bar" as const,
        label: "Predicted",
        data: predicted,
        backgroundColor: isLight
          ? "rgba(14,116,144,0.9)"
          : "rgba(99,102,241,0.9)",
      },
      {
        type: "line" as const,
        label: "Observed",
        data: observed,
        borderColor: isLight ? "rgba(6,78,59,0.9)" : "rgba(14,116,144,0.9)",
        tension: 0.3,
        fill: false,
        pointRadius: 3,
      },
    ],
  };

  const options: any = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
        labels: { color: isLight ? "#0f172a" : "#E6EEF3" },
      },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: isLight ? "#0f172a" : "#E6EEF3" },
      },
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          color: isLight ? "#0f172a" : "#E6EEF3",
          callback: (v: any) => Number(v).toFixed(2),
        },
        grid: {
          color: isLight ? "rgba(15,23,42,0.06)" : "rgba(255,255,255,0.06)",
        },
      },
    },
  };

  return (
    <div>
      <Bar data={data as any} options={options as any} />
    </div>
  );
}
