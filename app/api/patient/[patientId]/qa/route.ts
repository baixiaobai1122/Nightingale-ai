import { type NextRequest, NextResponse } from "next/server"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function GET(request: NextRequest, { params }: { params: { patientId: string } }) {
  try {
    const patientId = params.patientId
    const { searchParams } = new URL(request.url)
    const query = searchParams.get("q") || ""

    const response = await fetch(`${API_BASE_URL}/patient/${patientId}/qa?q=${encodeURIComponent(query)}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json({ error: data.detail || "Patient QA failed" }, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Patient QA API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
