"use client";

import { Chart } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
  BarController,
  LineController,
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
  Legend,
);

interface Props {
  prob_true: number[];
  prob_pred: number[];
  medicalLight?: boolean;
}

export default function ReliabilityChart({
  prob_true,
  prob_pred,
  medicalLight,
}: Props) {
  const isLight = !!medicalLight;
  const labels = prob_pred.map((_, i) => `Bin ${i + 1}`);

  const chartData = {
    labels,
    datasets: [
      {
        type: "bar" as const,
        label: "Predicted",
        data: prob_pred.map((v) => v * 100),
        backgroundColor: isLight
          ? "rgba(6,78,59,0.85)"
          : "rgba(14, 116, 144, 0.85)",
      },
      {
        type: "line" as const,
        label: "Observed (true)",
        data: prob_true.map((v) => v * 100),
        borderColor: isLight ? "rgba(6,78,59,0.9)" : "rgba(14, 116, 144, 0.9)",
        borderWidth: 2,
        fill: false,
        pointRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
        labels: { color: isLight ? "#374151" : "#D1D5DB", font: { size: 12 } },
      },
      tooltip: {
        callbacks: {
          label: (ctx: any) =>
            `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Bins",
          color: isLight ? "#374151" : "#9CA3AF",
        },
        ticks: { color: isLight ? "#374151" : "#9CA3AF", font: { size: 12 } },
        grid: {
          color: isLight ? "rgba(15,23,42,0.04)" : "rgba(148,163,184,0.06)",
        },
      },
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: "Probability (%)",
          color: isLight ? "#374151" : "#9CA3AF",
        },
        ticks: {
          callback: (v: any) => `${v}%`,
          color: isLight ? "#374151" : "#9CA3AF",
          font: { size: 12 },
        },
        grid: {
          color: isLight ? "rgba(15,23,42,0.04)" : "rgba(148,163,184,0.06)",
        },
      },
    },
  };

  return (
    <div className="w-full h-64">
      <Chart type="bar" data={chartData} options={options} />
    </div>
  );
}
