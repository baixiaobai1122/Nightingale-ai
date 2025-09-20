import { type NextRequest, NextResponse } from "next/server"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function GET(request: NextRequest, { params }: { params: { summaryId: string } }) {
  try {
    const summaryId = params.summaryId

    const response = await fetch(`${API_BASE_URL}/summary/${summaryId}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json({ error: data.detail || "Summary fetch failed" }, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Summary API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
