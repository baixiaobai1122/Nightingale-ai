import { type NextRequest, NextResponse } from "next/server"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch(`${API_BASE_URL}/consent/record`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        patient_name: body.patient_name,
        consent: body.consent,
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json({ error: data.detail || "Consent recording failed" }, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Consent API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
