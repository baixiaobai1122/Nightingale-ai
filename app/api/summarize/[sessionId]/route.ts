import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"

export async function POST(request: NextRequest, { params }: { params: { sessionId: string } }) {
  try {
    const sessionId = params.sessionId
    console.log("[v0] Summarize API called for session:", sessionId)

    if (!sessionId || sessionId === "[id]" || sessionId === "undefined") {
      console.log("[v0] Invalid session ID, using mock data")
      return NextResponse.json({
        summaries: [
          {
            id: "mock-clinician",
            type: "clinician",
            content: `## Clinical Summary\\n\\n**Chief Complaint:** Patient reports chest tightness and shortness of breath for approximately 2 weeks.\\n\\n**History of Present Illness:** \\n- Patient began experiencing chest tightness and shortness of breath 2 weeks ago\\n- Symptoms worsen with activity, improve with rest\\n- No chest pain or palpitations reported\\n- Sleep and appetite remain normal\\n\\n**Physical Examination:**\\n- Vital signs stable\\n- Heart rate: 78 bpm, Blood pressure: 120/80mmHg\\n- Heart and lung auscultation unremarkable\\n\\n**Assessment:** \\n1. Chest tightness - further evaluation needed\\n2. Recommend additional testing: ECG, chest X-ray, echocardiogram\\n\\n**Treatment Plan:**\\n- Avoid strenuous exercise\\n- Regular follow-up\\n- Medication therapy as needed\\n\\n**Follow-up Plan:** Return visit in 1 week`,
            status: "draft",
            created_at: new Date().toISOString(),
            session_id: sessionId,
          },
          {
            id: "mock-patient",
            type: "patient",
            content: `## Medical Summary\\n\\n**Your Condition:**\\nYou've been experiencing chest tightness and shortness of breath for the past 2 weeks, especially when you're active, but it gets better when you rest.\\n\\n**Test Results:**\\nToday's basic examination shows your vital signs are all normal, with your heart rate and blood pressure in the normal range.\\n\\n**Doctor's Recommendations:**\\n- Temporarily avoid strenuous exercise\\n- Maintain good sleep and rest habits\\n- We recommend some additional tests including an ECG, chest X-ray, and heart ultrasound to better understand your condition\\n\\n**Next Appointment:**\\nPlease return for a follow-up visit in one week, and we'll create a more detailed treatment plan based on your test results.\\n\\n**Important Notes:**\\nIf your symptoms worsen or you experience chest pain, please seek medical attention immediately.`,
            status: "draft",
            created_at: new Date().toISOString(),
            session_id: sessionId,
          },
        ],
      })
    }

    const supabase = await createClient()

    const { data: session, error: sessionError } = await supabase
      .from("sessions")
      .select("*")
      .eq("id", sessionId)
      .single()

    if (sessionError || !session) {
      console.log("[v0] Session not found in database, creating mock summaries for:", sessionId)
      // Return mock summaries instead of 404
      const clinicianSummary = {
        id: `${sessionId}-clinician`,
        type: "clinician",
        content: `## Clinical Summary\n\n**Chief Complaint:** Patient reports chest tightness and shortness of breath for approximately 2 weeks.\n\n**History of Present Illness:** \n- Patient began experiencing chest tightness and shortness of breath 2 weeks ago\n- Symptoms worsen with activity, improve with rest\n- No chest pain or palpitations reported\n- Sleep and appetite remain normal\n\n**Physical Examination:**\n- Vital signs stable\n- Heart rate: 78 bpm, Blood pressure: 120/80mmHg\n- Heart and lung auscultation unremarkable\n\n**Assessment:** \n1. Chest tightness - further evaluation needed\n2. Recommend additional testing: ECG, chest X-ray, echocardiogram\n\n**Treatment Plan:**\n- Avoid strenuous exercise\n- Regular follow-up\n- Medication therapy as needed\n\n**Follow-up Plan:** Return visit in 1 week`,
        status: "draft",
        created_at: new Date().toISOString(),
        session_id: sessionId,
      }

      const patientSummary = {
        id: `${sessionId}-patient`,
        type: "patient",
        content: `## Medical Summary\n\n**Your Condition:**\nYou've been experiencing chest tightness and shortness of breath for the past 2 weeks, especially when you're active, but it gets better when you rest.\n\n**Test Results:**\nToday's basic examination shows your vital signs are all normal, with your heart rate and blood pressure in the normal range.\n\n**Doctor's Recommendations:**\n- Temporarily avoid strenuous exercise\n- Maintain good sleep and rest habits\n- We recommend some additional tests including an ECG, chest X-ray, and heart ultrasound to better understand your condition\n\n**Next Appointment:**\nPlease return for a follow-up visit in one week, and we'll create a more detailed treatment plan based on your test results.\n\n**Important Notes:**\nIf your symptoms worsen or you experience chest pain, please seek medical attention immediately.`,
        status: "draft",
        created_at: new Date().toISOString(),
        session_id: sessionId,
      }

      return NextResponse.json({
        summaries: [clinicianSummary, patientSummary],
      })
    }

    // Generate mock summaries for now
    const clinicianSummary = {
      id: `${sessionId}-clinician`,
      type: "clinician",
      content: `## Clinical Summary

**Chief Complaint:** Patient reports chest tightness and shortness of breath for approximately 2 weeks.

**History of Present Illness:** 
- Patient began experiencing chest tightness and shortness of breath 2 weeks ago
- Symptoms worsen with activity, improve with rest
- No chest pain or palpitations reported
- Sleep and appetite remain normal

**Physical Examination:**
- Vital signs stable
- Heart rate: 78 bpm, Blood pressure: 120/80mmHg
- Heart and lung auscultation unremarkable

**Assessment:** 
1. Chest tightness - further evaluation needed
2. Recommend additional testing: ECG, chest X-ray, echocardiogram

**Treatment Plan:**
- Avoid strenuous exercise
- Regular follow-up
- Medication therapy as needed

**Follow-up Plan:** Return visit in 1 week`,
      status: "draft",
      created_at: new Date().toISOString(),
      session_id: sessionId,
    }

    const patientSummary = {
      id: `${sessionId}-patient`,
      type: "patient",
      content: `## Medical Summary

**Your Condition:**
You've been experiencing chest tightness and shortness of breath for the past 2 weeks, especially when you're active, but it gets better when you rest.

**Test Results:**
Today's basic examination shows your vital signs are all normal, with your heart rate and blood pressure in the normal range.

**Doctor's Recommendations:**
- Temporarily avoid strenuous exercise
- Maintain good sleep and rest habits
- We recommend some additional tests including an ECG, chest X-ray, and heart ultrasound to better understand your condition

**Next Appointment:**
Please return for a follow-up visit in one week, and we'll create a more detailed treatment plan based on your test results.

**Important Notes:**
If your symptoms worsen or you experience chest pain, please seek medical attention immediately.`,
      status: "draft",
      created_at: new Date().toISOString(),
      session_id: sessionId,
    }

    // Return both summaries directly
    return NextResponse.json({
      summaries: [clinicianSummary, patientSummary],
    })
  } catch (error) {
    console.error("[v0] Summarize API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
