"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import {
  FileText,
  User,
  Stethoscope,
  Clock,
  Shield,
  CheckCircle,
  Edit3,
  Save,
  Send,
  AlertCircle,
  ExternalLink,
} from "lucide-react"
import Link from "next/link"
import { useParams, useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"

interface Summary {
  id: string
  type: "clinician" | "patient"
  content: string
  status: "draft" | "approved"
  created_at: string
  session_id: string
}

export default function SessionSummaryPage() {
  const params = useParams()
  const router = useRouter()
  const sessionId = params.id as string
  const supabase = createClient()

  const [summaries, setSummaries] = useState<Summary[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editContent, setEditContent] = useState("")
  const [sessionInfo, setSessionInfo] = useState<any>(null)
  const [userRole, setUserRole] = useState<"patient" | "doctor" | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    checkUserRole()
    loadSessionInfo()
    generateSummaries()
  }, [sessionId])

  const checkUserRole = async () => {
    try {
      const {
        data: { user },
        error,
      } = await supabase.auth.getUser()

      if (!user) {
        router.push("/auth/login")
        return
      }

      const { data: profile, error: profileError } = await supabase
        .from("profiles")
        .select("role")
        .eq("id", user.id)
        .single()

      if (profileError || !profile) {
        router.push("/auth/login")
        return
      }

      setUserRole(profile.role)
      setIsLoading(false)
    } catch (error) {
      console.error("Failed to check user role:", error)
      router.push("/auth/login")
    }
  }

  const loadSessionInfo = async () => {
    console.log("[v0] Loading session info for:", sessionId)
    setSessionInfo({
      id: sessionId,
      patient_name: "Patient",
      start_time: new Date().toISOString(),
      duration: "23 minutes",
      segments_count: 12,
    })
  }

  const generateSummaries = async () => {
    setIsGenerating(true)
    try {
      console.log("[v0] Generating summaries for session:", sessionId)
      const response = await fetch(`/api/summarize/${sessionId}`, {
        method: "POST",
      })

      if (response.ok) {
        const data = await response.json()
        console.log("[v0] Summaries generated:", data)
        setSummaries(data.summaries)
      } else {
        console.error("[v0] Failed to generate summaries:", response.status)
      }
    } catch (error) {
      console.error("[v0] Failed to generate summaries:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  const startEditing = (summary: Summary) => {
    setEditingId(summary.id)
    setEditContent(summary.content)
  }

  const saveEdit = async (summaryId: string) => {
    setSummaries((prev) => prev.map((s) => (s.id === summaryId ? { ...s, content: editContent } : s)))
    setEditingId(null)
    setEditContent("")
  }

  const approveSummary = async (summaryId: string) => {
    try {
      const response = await fetch(`/api/review/${summaryId}/approve`, {
        method: "POST",
      })

      if (response.ok) {
        setSummaries((prev) => prev.map((s) => (s.id === summaryId ? { ...s, status: "approved" } : s)))
      }
    } catch (error) {
      console.error("Failed to approve summary:", error)
    }
  }

  const renderSummaryContent = (content: string) => {
    return content.split("\n").map((line, index) => {
      // Handle segment references
      const parts = line.split(/(\[S\d+\])/g)

      // Check if line is a heading
      if (line.startsWith("## ")) {
        return (
          <h2 key={index} className="text-xl font-semibold mt-6 mb-3 text-foreground">
            {line.replace("## ", "")}
          </h2>
        )
      }

      // Check if line is a subheading
      if (line.startsWith("### ")) {
        return (
          <h3 key={index} className="text-lg font-medium mt-4 mb-2 text-foreground">
            {line.replace("### ", "")}
          </h3>
        )
      }

      // Handle bullet points
      if (line.startsWith("- ")) {
        return (
          <li key={index} className="ml-4 mb-1 list-disc list-inside">
            {renderInlineFormatting(line.replace("- ", ""), parts)}
          </li>
        )
      }

      // Handle numbered lists
      if (/^\d+\.\s/.test(line)) {
        return (
          <li key={index} className="ml-4 mb-1 list-decimal list-inside">
            {renderInlineFormatting(line.replace(/^\d+\.\s/, ""), parts)}
          </li>
        )
      }

      // Handle regular paragraphs
      if (line.trim()) {
        return (
          <p key={index} className="mb-2">
            {renderInlineFormatting(line, parts)}
          </p>
        )
      }

      // Empty line
      return <br key={index} />
    })
  }

  const renderInlineFormatting = (text: string, parts: string[]) => {
    return parts.map((part, partIndex) => {
      // Handle segment references
      if (part.match(/\[S\d+\]/)) {
        return (
          <Badge key={partIndex} variant="outline" className="mx-1 text-xs">
            <ExternalLink className="w-3 h-3 mr-1" />
            {part}
          </Badge>
        )
      }

      // Handle bold text **text**
      if (part.includes("**")) {
        const boldParts = part.split(/(\*\*[^*]+\*\*)/g)
        return boldParts.map((boldPart, boldIndex) => {
          if (boldPart.startsWith("**") && boldPart.endsWith("**")) {
            return (
              <strong key={`${partIndex}-${boldIndex}`} className="font-semibold">
                {boldPart.replace(/\*\*/g, "")}
              </strong>
            )
          }
          return <span key={`${partIndex}-${boldIndex}`}>{boldPart}</span>
        })
      }

      // Handle italic text *text*
      if (part.includes("*") && !part.includes("**")) {
        const italicParts = part.split(/(\*[^*]+\*)/g)
        return italicParts.map((italicPart, italicIndex) => {
          if (italicPart.startsWith("*") && italicPart.endsWith("*") && !italicPart.includes("**")) {
            return (
              <em key={`${partIndex}-${italicIndex}`} className="italic">
                {italicPart.replace(/\*/g, "")}
              </em>
            )
          }
          return <span key={`${partIndex}-${italicIndex}`}>{italicPart}</span>
        })
      }

      return <span key={partIndex}>{part}</span>
    })
  }

  const getCurrentSummary = () => {
    if (!userRole) return null
    return summaries.find((summary) =>
      userRole === "patient" ? summary.type === "patient" : summary.type === "clinician",
    )
  }

  const getSummaryTitle = () => {
    if (userRole === "patient") {
      return "Medical Summary"
    } else {
      return "Clinical Summary"
    }
  }

  const getBackLink = () => {
    if (userRole === "patient") {
      return "/patient/dashboard"
    } else {
      return "/doctor/dashboard"
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    )
  }

  const currentSummary = getCurrentSummary()

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href={getBackLink()} className="text-muted-foreground hover:text-foreground">
              ← Back to Dashboard
            </Link>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="text-success border-success/50">
              <Shield className="w-3 h-3 mr-1" />
              Privacy Protected
            </Badge>
            {sessionInfo && (
              <Badge variant="default">
                <Clock className="w-3 h-3 mr-1" />
                {sessionInfo.duration}
              </Badge>
            )}
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Session Info */}
          {sessionInfo && (
            <Card className="medical-gradient border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  {getSummaryTitle()} - {sessionInfo.patient_name}
                </CardTitle>
                <CardDescription>
                  Session time: {new Date(sessionInfo.start_time).toLocaleString()} • Duration: {sessionInfo.duration} •
                  Conversation segments: {sessionInfo.segments_count}
                </CardDescription>
              </CardHeader>
            </Card>
          )}

          {/* Generate Button */}
          {summaries.length === 0 && (
            <Card className="medical-gradient border-border/50">
              <CardContent className="pt-6">
                <div className="text-center space-y-4">
                  {isGenerating ? (
                    <>
                      <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto" />
                      <p className="text-muted-foreground">Generating AI summary...</p>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-12 h-12 text-muted-foreground mx-auto" />
                      <p className="text-muted-foreground">Ready to generate consultation summary</p>
                      <Button onClick={generateSummaries} size="lg">
                        <Send className="w-4 h-4 mr-2" />
                        Generate AI Summary
                      </Button>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {currentSummary && (
            <Card className="medical-gradient border-border/50">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    {userRole === "patient" ? <User className="w-5 h-5" /> : <Stethoscope className="w-5 h-5" />}
                    {getSummaryTitle()}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={currentSummary.status === "approved" ? "default" : "secondary"}
                      className={currentSummary.status === "approved" ? "bg-success" : ""}
                    >
                      {currentSummary.status === "approved" ? (
                        <CheckCircle className="w-3 h-3 mr-1" />
                      ) : (
                        <AlertCircle className="w-3 h-3 mr-1" />
                      )}
                      {currentSummary.status === "approved" ? "Approved" : "Pending Review"}
                    </Badge>
                  </div>
                </div>
                <CardDescription>
                  {userRole === "patient"
                    ? "Patient-friendly summary using clear, understandable language to explain the consultation process"
                    : "Structured clinical summary containing key medical information and diagnostic points"}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {editingId === currentSummary.id ? (
                  <div className="space-y-4">
                    <Textarea
                      value={editContent}
                      onChange={(e) => setEditContent(e.target.value)}
                      className="min-h-[300px] bg-background/50"
                    />
                    <div className="flex gap-2">
                      <Button onClick={() => saveEdit(currentSummary.id)}>
                        <Save className="w-4 h-4 mr-2" />
                        Save Changes
                      </Button>
                      <Button variant="outline" onClick={() => setEditingId(null)}>
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="prose prose-sm max-w-none bg-accent/30 p-4 rounded-lg border border-border/50">
                      {renderSummaryContent(currentSummary.content)}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        onClick={() => startEditing(currentSummary)}
                        disabled={currentSummary.status === "approved"}
                      >
                        <Edit3 className="w-4 h-4 mr-2" />
                        Edit Summary
                      </Button>
                      {userRole === "doctor" && currentSummary.status === "draft" && (
                        <Button onClick={() => approveSummary(currentSummary.id)}>
                          <CheckCircle className="w-4 h-4 mr-2" />
                          Approve Summary
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
