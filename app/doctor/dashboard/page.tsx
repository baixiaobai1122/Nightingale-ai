"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Stethoscope,
  Users,
  FileText,
  Filter,
  CheckCircle,
  AlertCircle,
  Calendar,
  User,
  Edit3,
  Eye,
  LogOut,
} from "lucide-react"
import Link from "next/link"
import { createClient } from "@/lib/supabase/client"
import { useRouter } from "next/navigation"

interface Session {
  id: string
  patient_name: string
  patient_id: string
  session_date: string
  status: "active" | "completed"
  summaries: {
    clinician_id?: string
    patient_id?: string
    clinician_status: "draft" | "approved"
    patient_status: "draft" | "approved"
  }[]
}

interface DoctorProfile {
  name: string
  staff_id: string
  total_patients: number
  pending_reviews: number
  completed_sessions: number
}

export default function DoctorDashboardPage() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [filteredSessions, setFilteredSessions] = useState<Session[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState("all")
  const [reviewFilter, setReviewFilter] = useState("all")
  const [isLoading, setIsLoading] = useState(true)
  const [doctorInfo, setDoctorInfo] = useState<DoctorProfile>({
    name: "Loading...",
    staff_id: "DOC001",
    total_patients: 45,
    pending_reviews: 8,
    completed_sessions: 156,
  })
  const [isLoadingProfile, setIsLoadingProfile] = useState(true)

  const router = useRouter()

  useEffect(() => {
    loadDoctorProfile()
    loadSessions()
  }, [])

  useEffect(() => {
    filterSessions()
  }, [sessions, searchQuery, statusFilter, reviewFilter])

  const loadDoctorProfile = async () => {
    setIsLoadingProfile(true)
    try {
      const supabase = createClient()

      // Get current user
      const {
        data: { user },
        error: userError,
      } = await supabase.auth.getUser()
      if (userError || !user) {
        console.error("Error getting user:", userError)
        router.push("/auth/login")
        return
      }

      // Get user profile from profiles table
      const { data: profile, error: profileError } = await supabase
        .from("profiles")
        .select("name, role")
        .eq("id", user.id)
        .single()

      if (profileError) {
        console.error("Error loading profile:", profileError)
        // Keep default values if profile not found
        return
      }

      if (profile) {
        setDoctorInfo((prev) => ({
          ...prev,
          name: profile.name || "Dr. User",
        }))
      }
    } catch (error) {
      console.error("Failed to load doctor profile:", error)
    } finally {
      setIsLoadingProfile(false)
    }
  }

  const loadSessions = async () => {
    setIsLoading(true)
    try {
      // Mock data - replace with real API call
      const mockSessions: Session[] = [
        {
          id: "sess_001",
          patient_name: "<NAME_1>",
          patient_id: "pat_001",
          session_date: "2024-01-15T09:30:00Z",
          status: "completed",
          summaries: [
            {
              clinician_id: "sum_001_clin",
              patient_id: "sum_001_pat",
              clinician_status: "draft",
              patient_status: "draft",
            },
          ],
        },
        {
          id: "sess_002",
          patient_name: "<NAME_2>",
          patient_id: "pat_002",
          session_date: "2024-01-15T14:15:00Z",
          status: "completed",
          summaries: [
            {
              clinician_id: "sum_002_clin",
              patient_id: "sum_002_pat",
              clinician_status: "approved",
              patient_status: "approved",
            },
          ],
        },
        {
          id: "sess_003",
          patient_name: "<NAME_3>",
          patient_id: "pat_003",
          session_date: "2024-01-16T10:00:00Z",
          status: "active",
          summaries: [],
        },
      ]
      setSessions(mockSessions)
    } catch (error) {
      console.error("Failed to load sessions:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const filterSessions = () => {
    let filtered = sessions

    // Search filter
    if (searchQuery.trim()) {
      filtered = filtered.filter(
        (session) =>
          session.patient_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          session.id.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    }

    // Status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter((session) => session.status === statusFilter)
    }

    // Review filter
    if (reviewFilter !== "all") {
      filtered = filtered.filter((session) => {
        if (reviewFilter === "pending") {
          return session.summaries.some((s) => s.clinician_status === "draft")
        } else if (reviewFilter === "approved") {
          return session.summaries.some((s) => s.clinician_status === "approved")
        }
        return true
      })
    }

    setFilteredSessions(filtered)
  }

  const getSessionStatusBadge = (session: Session) => {
    if (session.status === "active") {
      return (
        <Badge variant="default" className="bg-warning">
          In Progress
        </Badge>
      )
    }

    const hasPendingReview = session.summaries.some((s) => s.clinician_status === "draft")
    if (hasPendingReview) {
      return (
        <Badge variant="outline" className="border-destructive text-destructive">
          Pending Review
        </Badge>
      )
    }

    return (
      <Badge variant="default" className="bg-success">
        Completed
      </Badge>
    )
  }

  const handleApprove = async (summaryId: string) => {
    try {
      const response = await fetch(`/api/review/${summaryId}/approve`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access_token")}`,
        },
      })

      if (response.ok) {
        loadSessions() // Refresh data
      }
    } catch (error) {
      console.error("Failed to approve summary:", error)
    }
  }

  const handleLogout = async () => {
    try {
      const supabase = createClient()
      const { error } = await supabase.auth.signOut()

      if (error) {
        console.error("Error logging out:", error)
        return
      }

      // Redirect to login page after successful logout
      router.push("/auth/login")
    } catch (error) {
      console.error("Failed to logout:", error)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Stethoscope className="w-5 h-5 text-primary-foreground" />
            </div>
            <h1 className="text-xl font-semibold text-foreground">Clinical Workstation</h1>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="text-primary border-primary/50">
              Doctor Portal
            </Badge>
            <span className="text-sm text-muted-foreground">
              Welcome, {isLoadingProfile ? "Loading..." : doctorInfo.name}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={handleLogout}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground bg-transparent"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="warm-card border-border/50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
                <Users className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{doctorInfo.total_patients}</div>
                <p className="text-xs text-muted-foreground">Under your care</p>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Pending Reviews</CardTitle>
                <AlertCircle className="h-4 w-4 text-warning" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-warning">{doctorInfo.pending_reviews}</div>
                <p className="text-xs text-muted-foreground">Summaries to review</p>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Completed Sessions</CardTitle>
                <CheckCircle className="h-4 w-4 text-success" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{doctorInfo.completed_sessions}</div>
                <p className="text-xs text-muted-foreground">Total consultations</p>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Today's Sessions</CardTitle>
                <Calendar className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">6</div>
                <p className="text-xs text-muted-foreground">Scheduled appointments</p>
              </CardContent>
            </Card>
          </div>

          {/* Filters and Search */}
          <Card className="warm-card border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Filter className="w-5 h-5" />
                Session Management
              </CardTitle>
              <CardDescription>
                View and manage consultation sessions by patient, time, or review status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex-1">
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search patient name or session ID..."
                    className="bg-background/50"
                  />
                </div>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-full sm:w-40">
                    <SelectValue placeholder="Session Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="active">In Progress</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={reviewFilter} onValueChange={setReviewFilter}>
                  <SelectTrigger className="w-full sm:w-40">
                    <SelectValue placeholder="Review Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Reviews</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="approved">Approved</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Sessions List */}
          <Card className="warm-card border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Session Records
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
                  <p className="text-muted-foreground">Loading sessions...</p>
                </div>
              ) : filteredSessions.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">No matching session records found</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {filteredSessions.map((session) => (
                    <div key={session.id} className="p-4 rounded-lg bg-accent/30 border border-border/50">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                            <User className="w-5 h-5 text-primary" />
                          </div>
                          <div>
                            <h3 className="font-medium">{session.patient_name}</h3>
                            <p className="text-sm text-muted-foreground">
                              {new Date(session.session_date).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">{getSessionStatusBadge(session)}</div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm" asChild>
                            <Link href={`/session/${session.id}/summary`}>
                              <Eye className="w-4 h-4 mr-1" />
                              View Summary
                            </Link>
                          </Button>
                          {session.summaries.some((s) => s.clinician_status === "draft") && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() =>
                                session.summaries[0]?.clinician_id && handleApprove(session.summaries[0].clinician_id)
                              }
                            >
                              <CheckCircle className="w-4 h-4 mr-1" />
                              Approve Summary
                            </Button>
                          )}
                          <Button variant="outline" size="sm" asChild>
                            <Link href={`/session/${session.id}/edit`}>
                              <Edit3 className="w-4 h-4 mr-1" />
                              Edit Notes
                            </Link>
                          </Button>
                        </div>
                        <p className="text-xs text-muted-foreground">Session ID: {session.id}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <div className="flex gap-4">
            <Button asChild className="flex-1">
              <Link href="/session/new">
                <Calendar className="w-4 h-4 mr-2" />
                Start New Session
              </Link>
            </Button>
            <Button variant="outline" asChild className="flex-1 bg-transparent">
              <Link href="/session/sess_001/summary">
                <Users className="w-4 h-4 mr-2" />
                Patient Consent Management
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
