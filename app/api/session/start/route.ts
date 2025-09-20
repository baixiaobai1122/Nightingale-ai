import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { patient_id } = body

    if (!patient_id) {
      return NextResponse.json({ error: "Patient ID is required" }, { status: 400 })
    }

    const supabase = await createClient()

    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

    const { data: sessionData, error: sessionError } = await supabase
      .from("sessions")
      .insert({
        id: sessionId,
        patient_id: patient_id,
        status: "active",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      .select()
      .single()

    if (sessionError) {
      console.error("Database error:", sessionError)
      return NextResponse.json({ error: "Failed to create session" }, { status: 500 })
    }

    return NextResponse.json({
      session_id: sessionId,
      patient_id: patient_id,
      status: "active",
      created_at: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Session start API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
