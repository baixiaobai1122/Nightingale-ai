"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Stethoscope, User, Shield, UserCheck } from "lucide-react"
import { useRouter } from "next/navigation"

export default function RoleSelectPage() {
  const router = useRouter()
  const [selectedRole, setSelectedRole] = useState<"patient" | "doctor" | null>(null)

  const handleRoleSelect = (role: "patient" | "doctor") => {
    setSelectedRole(role)
    localStorage.setItem("selectedRole", role)

    if (role === "doctor") {
      router.push("/auth/signup")
    } else if (role === "patient") {
      router.push("/patient/consent")
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-medical-warm to-background flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-lg bg-primary flex items-center justify-center">
              <Stethoscope className="w-7 h-7 text-primary-foreground" />
            </div>
            <h1 className="text-3xl font-bold text-foreground">Nightingale AI</h1>
          </div>
          <h2 className="text-2xl font-semibold mb-4">Welcome to secure medical AI</h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Please select your role to access the appropriate interface designed for your needs.
          </p>
        </div>

        {/* Role Selection Cards */}
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          {/* Patient Card */}
          <Card
            className="warm-card border-2 hover:border-primary/50 transition-all duration-200 cursor-pointer group h-full flex flex-col"
            onClick={() => handleRoleSelect("patient")}
          >
            <CardHeader className="text-center pb-4">
              <div className="w-20 h-20 rounded-full bg-success/20 flex items-center justify-center mx-auto mb-4 group-hover:bg-success/30 transition-colors">
                <User className="w-10 h-10 text-success" />
              </div>
              <CardTitle className="text-2xl">I'm a Patient</CardTitle>
              <CardDescription className="text-base">
                Access your medical summaries, ask questions about your care, and manage your health information.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <div className="space-y-3 text-sm text-muted-foreground flex-1">
                <div className="flex items-center gap-3">
                  <Shield className="w-4 h-4 text-success" />
                  <span>View your personalized health summaries</span>
                </div>
                <div className="flex items-center gap-3">
                  <Shield className="w-4 h-4 text-success" />
                  <span>Ask questions about your medical visits</span>
                </div>
                <div className="flex items-center gap-3">
                  <Shield className="w-4 h-4 text-success" />
                  <span>Simple, secure account verification</span>
                </div>
                <div className="flex items-center gap-3">
                  <Shield className="w-4 h-4 text-success" />
                  <span>Your privacy is completely protected</span>
                </div>
              </div>
              <Button className="w-full mt-6" size="lg">
                Continue as Patient
              </Button>
            </CardContent>
          </Card>

          {/* Doctor Card */}
          <Card
            className="warm-card border-2 hover:border-primary/50 transition-all duration-200 cursor-pointer group h-full flex flex-col"
            onClick={() => handleRoleSelect("doctor")}
          >
            <CardHeader className="text-center pb-4">
              <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4 group-hover:bg-primary/30 transition-colors">
                <Stethoscope className="w-10 h-10 text-primary" />
              </div>
              <CardTitle className="text-2xl">I'm a Healthcare Provider</CardTitle>
              <CardDescription className="text-base">
                Manage patient sessions, review summaries, and access clinical documentation tools.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <div className="space-y-3 text-sm text-muted-foreground flex-1">
                <div className="flex items-center gap-3">
                  <UserCheck className="w-4 h-4 text-primary" />
                  <span>Manage all patient consultations</span>
                </div>
                <div className="flex items-center gap-3">
                  <UserCheck className="w-4 h-4 text-primary" />
                  <span>Review and approve medical summaries</span>
                </div>
                <div className="flex items-center gap-3">
                  <UserCheck className="w-4 h-4 text-primary" />
                  <span>Enhanced security with staff ID verification</span>
                </div>
                <div className="flex items-center gap-3">
                  <UserCheck className="w-4 h-4 text-primary" />
                  <span>Professional clinical interface</span>
                </div>
              </div>
              <Button className="w-full mt-6" size="lg">
                Continue as Healthcare Provider
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Privacy Notice */}
        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
            <Shield className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary font-medium">All data is encrypted and privacy-protected</span>
          </div>
        </div>
      </div>
    </div>
  )
}
