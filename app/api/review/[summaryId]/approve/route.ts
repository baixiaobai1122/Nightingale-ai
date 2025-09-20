import { type NextRequest, NextResponse } from "next/server"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function POST(request: NextRequest, { params }: { params: { summaryId: string } }) {
  try {
    const summaryId = params.summaryId

    const response = await fetch(`${API_BASE_URL}/review/${summaryId}/approve`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json({ error: data.detail || "Approval failed" }, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Approval API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
