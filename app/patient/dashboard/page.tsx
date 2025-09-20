"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { User, MessageCircle, FileText, Clock, Search, Shield, Heart, Calendar, LogOut } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"

export default function PatientDashboardPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [recentSummaries, setRecentSummaries] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [user, setUser] = useState<any>(null)
  const [profile, setProfile] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    const checkAuth = async () => {
      console.log("[v0] Checking authentication in patient dashboard")

      const {
        data: { user },
        error,
      } = await supabase.auth.getUser()
      console.log("[v0] User data:", user)
      console.log("[v0] Auth error:", error)

      if (!user) {
        console.log("[v0] No user found, redirecting to login")
        router.push("/auth/login")
        return
      }

      const { data: profileData, error: profileError } = await supabase
        .from("profiles")
        .select("*")
        .eq("id", user.id)
        .single()

      console.log("[v0] Profile data:", profileData)
      console.log("[v0] Profile error:", profileError)

      if (profileError || !profileData) {
        console.log("[v0] No profile found, redirecting to login")
        router.push("/auth/login")
        return
      }

      if (profileData.role !== "patient") {
        console.log("[v0] User is not a patient, redirecting to doctor dashboard")
        router.push("/doctor/dashboard")
        return
      }

      console.log("[v0] Authentication successful, setting user data")
      setUser(user)
      setProfile(profileData)
      setIsLoading(false)
      loadRecentSummaries()
    }

    checkAuth()
  }, [])

  const patientInfo = {
    id: user?.id || "1",
    name: profile?.name || "Patient",
    lastVisit: "2024-01-15T09:30:00Z",
    totalVisits: 8,
    approvedSummaries: 12,
  }

  const loadRecentSummaries = async () => {
    setRecentSummaries([
      {
        id: "sum_001",
        session_date: "2024-01-15T09:30:00Z",
        title: "Routine Check-up and Health Consultation",
        status: "approved",
        key_points: ["Blood pressure normal", "Recommend increased exercise", "Good nutritional status"],
      },
      {
        id: "sum_002",
        session_date: "2024-01-08T14:15:00Z",
        title: "Follow-up and Medication Adjustment",
        status: "approved",
        key_points: ["Medication working well", "Symptoms improving", "Continue current treatment plan"],
      },
    ])
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    try {
      const response = await fetch(`/api/patient/${patientInfo.id}/qa?q=${encodeURIComponent(searchQuery)}`)
      if (response.ok) {
        const data = await response.json()
        setSearchResults(data.results || [])
      }
    } catch (error) {
      console.error("Search failed:", error)
    } finally {
      setIsSearching(false)
    }
  }

  const handleLogout = async () => {
    console.log("[v0] Logging out user")
    await supabase.auth.signOut()
    router.push("/auth/login")
  }

  const renderSearchResults = () => {
    if (searchResults.length === 0) return null

    return (
      <Card className="medical-gradient border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="w-5 h-5" />
            Search Results
          </CardTitle>
          <CardDescription>Found the following relevant information based on your medical history</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {searchResults.map((result, index) => (
              <div key={index} className="p-4 rounded-lg bg-accent/50 border border-border/50">
                <div className="flex items-start justify-between mb-2">
                  <p className="text-sm font-medium">{result.snippet}</p>
                  <div className="flex gap-1">
                    {result.citations?.map((citation: string, idx: number) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {citation}
                      </Badge>
                    ))}
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  Type: {result.type === "patient" ? "Patient Record" : "Doctor Record"}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Heart className="w-5 h-5 text-primary-foreground" />
            </div>
            <h1 className="text-xl font-semibold text-foreground">My Health Records</h1>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="text-success border-success/50">
              <Shield className="w-3 h-3 mr-1" />
              Privacy Protected
            </Badge>
            <span className="text-sm text-muted-foreground">Welcome, {profile?.name}</span>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              <LogOut className="w-4 h-4 mr-2" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Patient Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="medical-gradient border-border/50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Last Visit</CardTitle>
                <Calendar className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{new Date(patientInfo.lastVisit).toLocaleDateString()}</div>
                <p className="text-xs text-muted-foreground">Most recent appointment</p>
              </CardContent>
            </Card>

            <Card className="medical-gradient border-border/50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Visits</CardTitle>
                <User className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{patientInfo.totalVisits}</div>
                <p className="text-xs text-muted-foreground">Total appointments</p>
              </CardContent>
            </Card>

            <Card className="medical-gradient border-border/50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Health Records</CardTitle>
                <FileText className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{patientInfo.approvedSummaries}</div>
                <p className="text-xs text-muted-foreground">Available records</p>
              </CardContent>
            </Card>
          </div>

          {/* Search Section */}
          <Card className="medical-gradient border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageCircle className="w-5 h-5" />
                Ask About Your Health Records
              </CardTitle>
              <CardDescription>You can search and ask questions about your medical history</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="e.g., How is my blood pressure? What were my last test results?"
                  className="bg-background/50"
                  onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                />
                <Button onClick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
                  {isSearching ? (
                    <div className="animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
                  ) : (
                    <Search className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Search Results */}
          {renderSearchResults()}

          {/* Recent Summaries */}
          <Card className="medical-gradient border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Recent Medical Records
              </CardTitle>
              <CardDescription>Your recent visit summaries and health recommendations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentSummaries.map((summary) => (
                  <div key={summary.id} className="p-4 rounded-lg bg-accent/50 border border-border/50">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-medium text-sm">{summary.title}</h3>
                        <p className="text-xs text-muted-foreground">
                          {new Date(summary.session_date).toLocaleDateString()}
                        </p>
                      </div>
                      <Badge variant="default" className="bg-success">
                        Completed
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <p className="text-xs text-muted-foreground mb-2">Key Points:</p>
                      <ul className="space-y-1">
                        {summary.key_points.map((point: string, index: number) => (
                          <li key={index} className="text-sm flex items-center gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                            {point}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <Button variant="outline" size="sm" className="mt-3 bg-transparent" asChild>
                      <Link href={`/patient/summary/${summary.id}`}>View Full Record</Link>
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="medical-gradient border-border/50">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>Common features and services</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button variant="outline" className="h-auto p-4 justify-start bg-transparent" asChild>
                  <Link href="/session/sess_001/summary">
                    <FileText className="w-5 h-5 mr-3" />
                    <div className="text-left">
                      <div className="font-medium">Summary Records</div>
                      <div className="text-sm text-muted-foreground">View your visit summaries</div>
                    </div>
                  </Link>
                </Button>
                <Button variant="outline" className="h-auto p-4 justify-start bg-transparent" asChild>
                  <Link href="/session/new">
                    <MessageCircle className="w-5 h-5 mr-3" />
                    <div className="text-left">
                      <div className="font-medium">Start New Session</div>
                      <div className="text-sm text-muted-foreground">Record new health consultation</div>
                    </div>
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
