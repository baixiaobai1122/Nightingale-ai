"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import {
  Mic,
  MicOff,
  Play,
  Pause,
  Square,
  Clock,
  User,
  Shield,
  FileText,
  Send,
  Volume2,
  AlertCircle,
} from "lucide-react"
import Link from "next/link"

export default function NewSessionPage() {
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [sessionStarted, setSessionStarted] = useState(false)
  const [duration, setDuration] = useState(0)
  const [patientId, setPatientId] = useState("")
  const [patientName, setPatientName] = useState("")
  const [manualText, setManualText] = useState("")
  const [segments, setSegments] = useState<
    Array<{
      id: string
      text: string
      timestamp: string
      duration: number
      isRedacted: boolean
    }>
  >([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)

  const intervalRef = useRef<NodeJS.Timeout>()
  const mediaRecorderRef = useRef<MediaRecorder>()

  // Timer effect
  useEffect(() => {
    if (sessionStarted && !isPaused) {
      intervalRef.current = setInterval(() => {
        setDuration((prev) => prev + 1)
      }, 1000)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [sessionStarted, isPaused])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  const startSession = async () => {
    if (!patientId.trim()) {
      alert("Please enter Patient ID")
      return
    }

    try {
      // Call backend API to start session
      const response = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: patientId }),
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentSessionId(data.session_id)
        setSessionStarted(true)
        setDuration(0)
      }
    } catch (error) {
      console.error("Failed to start session:", error)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorderRef.current = new MediaRecorder(stream)

      mediaRecorderRef.current.ondataavailable = (event) => {
        // Handle audio data - would send to ASR service
        console.log("Audio data available:", event.data)
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)
    } catch (error) {
      console.error("Failed to start recording:", error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop())
    }
    setIsRecording(false)
  }

  const addManualSegment = () => {
    if (!manualText.trim() || !currentSessionId) return

    const newSegment = {
      id: `seg_${Date.now()}`,
      text: manualText,
      timestamp: new Date().toISOString(),
      duration: 0,
      isRedacted: manualText.includes("<") && manualText.includes(">"),
    }

    setSegments((prev) => [...prev, newSegment])
    setManualText("")

    // Send to backend
    fetch("/api/asr/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        segments: [
          {
            text: manualText,
            start_ms: duration * 1000,
            end_ms: (duration + 5) * 1000,
          },
        ],
      }),
    })
  }

  const endSession = () => {
    if (isRecording) {
      stopRecording()
    }
    setSessionStarted(false)
    setIsPaused(false)
    // Redirect to summary generation
    if (currentSessionId) {
      window.location.href = `/session/${currentSessionId}/summary`
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/patient/dashboard" className="text-muted-foreground hover:text-foreground">
              ← Back to Dashboard
            </Link>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="text-success border-success/50">
              <Shield className="w-3 h-3 mr-1" />
              Privacy Protected
            </Badge>
            {sessionStarted && (
              <Badge variant="default" className="bg-primary">
                <Clock className="w-3 h-3 mr-1" />
                {formatTime(duration)}
              </Badge>
            )}
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Session Setup */}
          {!sessionStarted && (
            <Card className="medical-gradient border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="w-5 h-5" />
                  Start New Consultation
                </CardTitle>
                <CardDescription>Enter patient information to begin recording the consultation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="patientId">Patient ID *</Label>
                    <Input
                      id="patientId"
                      value={patientId}
                      onChange={(e) => setPatientId(e.target.value)}
                      placeholder="Enter Patient ID"
                      className="bg-background/50"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="patientName">Patient Name (Optional)</Label>
                    <Input
                      id="patientName"
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      placeholder="Will be automatically redacted"
                      className="bg-background/50"
                    />
                  </div>
                </div>
                <Button onClick={startSession} className="w-full" size="lg">
                  Start Consultation Recording
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Active Session */}
          {sessionStarted && (
            <>
              {/* Recording Controls */}
              <Card className="medical-gradient border-border/50">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Volume2 className="w-5 h-5" />
                      Audio Recording
                    </span>
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-3 h-3 rounded-full ${isRecording ? "bg-destructive animate-pulse" : "bg-muted"}`}
                      />
                      <span className="text-sm text-muted-foreground">{isRecording ? "Recording" : "Paused"}</span>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center gap-4">
                    {!isRecording ? (
                      <Button onClick={startRecording} size="lg" className="bg-destructive hover:bg-destructive/90">
                        <Mic className="w-5 h-5 mr-2" />
                        Start Recording
                      </Button>
                    ) : (
                      <Button onClick={stopRecording} size="lg" variant="outline">
                        <MicOff className="w-5 h-5 mr-2" />
                        Stop Recording
                      </Button>
                    )}
                    <Button onClick={() => setIsPaused(!isPaused)} size="lg" variant="outline">
                      {isPaused ? <Play className="w-5 h-5 mr-2" /> : <Pause className="w-5 h-5 mr-2" />}
                      {isPaused ? "Resume" : "Pause"}
                    </Button>
                    <Button onClick={endSession} size="lg" variant="destructive">
                      <Square className="w-5 h-5 mr-2" />
                      End Consultation
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Manual Text Input */}
              <Card className="medical-gradient border-border/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    Manual Text Input
                  </CardTitle>
                  <CardDescription>
                    You can manually input conversation content, the system will automatically handle privacy redaction
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <Textarea
                      value={manualText}
                      onChange={(e) => setManualText(e.target.value)}
                      placeholder="Enter conversation content..."
                      className="min-h-[100px] bg-background/50"
                    />
                    <Button onClick={addManualSegment} disabled={!manualText.trim()}>
                      <Send className="w-4 h-4 mr-2" />
                      Add Text Segment
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Session Timeline */}
              <Card className="medical-gradient border-border/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Consultation Timeline
                  </CardTitle>
                  <CardDescription>Real-time display of conversation segments and redaction status</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {segments.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <AlertCircle className="w-8 h-8 mx-auto mb-2" />
                        No conversation segments yet
                      </div>
                    ) : (
                      segments.map((segment, index) => (
                        <div
                          key={segment.id}
                          className="flex items-start gap-3 p-3 rounded-lg bg-accent/50 border border-border/50"
                        >
                          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-xs font-medium">
                            {index + 1}
                          </div>
                          <div className="flex-1 space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-muted-foreground">
                                {new Date(segment.timestamp).toLocaleTimeString()}
                              </span>
                              {segment.isRedacted && (
                                <Badge variant="secondary" className="text-xs">
                                  <Shield className="w-3 h-3 mr-1" />
                                  Redacted
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm">{segment.text}</p>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
