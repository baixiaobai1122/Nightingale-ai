"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"

export default function AuthCallbackPage() {
  const router = useRouter()

  useEffect(() => {
    const handleAuthCallback = async () => {
      const supabase = createClient()

      const { data, error } = await supabase.auth.getSession()

      if (error) {
        console.error("Auth callback error:", error)
        router.push("/auth/login?error=Authentication failed")
        return
      }

      if (data.session?.user) {
        // Get user profile to determine role
        const { data: profile } = await supabase.from("profiles").select("role").eq("id", data.session.user.id).single()

        // Redirect based on user role
        if (profile?.role === "doctor") {
          router.push("/doctor/dashboard")
        } else {
          router.push("/patient/dashboard")
        }
      } else {
        router.push("/auth/login")
      }
    }

    handleAuthCallback()
  }, [router])

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
        <p className="text-muted-foreground">Completing authentication...</p>
      </div>
    </div>
  )
}
