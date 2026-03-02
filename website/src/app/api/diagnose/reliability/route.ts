import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:7860";

    // Try configured backend, fallback to local; if both fail, return a safe fallback
    let response: Response | null = null;
    let backendError: string | null = null;
    try {
      response = await fetch(`${BACKEND_URL}/diagnose/reliability`);
      if (!response.ok) {
        // attempt to read backend error body for debugging
        let txt = await response.text().catch(() => "");
        backendError = `Primary ${BACKEND_URL} responded ${response.status}: ${txt}`;
        throw new Error(backendError);
      }
    } catch (e) {
      console.warn(
        `Primary backend ${BACKEND_URL} failed; trying http://127.0.0.1:7860`,
        e,
      );
      try {
        response = await fetch(`http://127.0.0.1:7860/diagnose/reliability`);
        if (!response.ok) {
          let txt = await response.text().catch(() => "");
          backendError = `Local backend responded ${response.status}: ${txt}`;
          throw new Error(backendError);
        }
      } catch (e2) {
        console.warn("Local backend fetch failed", e2);
        // Return a harmless fallback so the UI can render with sample reliability data
        const fallback = {
          bins: Array.from({ length: 10 }, (_, i) => (i + 0.5) / 10),
          prob_pred: [0.05, 0.1, 0.12, 0.1, 0.1, 0.1, 0.12, 0.1, 0.08, 0.13],
          prob_true: [
            0.04, 0.09, 0.1, 0.11, 0.09, 0.11, 0.13, 0.12, 0.08, 0.13,
          ],
          brier_score: 0.12,
          confusion_matrix: [[0]],
          class_labels: ["class0", "class1"],
          _fallback: true,
          _backend_error: backendError,
        };
        return NextResponse.json(fallback);
      }
    }

    if (!response) {
      return NextResponse.json(
        { error: "No response from backend" },
        { status: 502 }
      );
    }

    try {
      const data = await response.json();
      return NextResponse.json(data);
    } catch (err) {
      console.error("Failed to parse backend reliability JSON", err);
      // include backend error text if available
      return NextResponse.json(
        { error: "Invalid backend response", _backend_error: backendError },
        { status: 502 },
      );
    }
  } catch (err) {
    console.error(err);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
