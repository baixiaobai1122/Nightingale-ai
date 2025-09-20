"use client"

import { useState, useCallback } from "react"

interface ApiResponse<T = any> {
  data?: T
  error?: string
  status: number
}

interface LoginCredentials {
  username: string
  password: string
}

interface SignupData {
  name: string
  password: string
}

interface ConsentData {
  patient_name: string
  consent: boolean
}

interface SessionData {
  patient_id: number
}

interface SegmentData {
  session_id: number
  segments: Array<{
    text: string
    start_ms: number
    end_ms: number
  }>
}

export function useApiClient() {
  const [loading, setLoading] = useState(false)
  const [token, setToken] = useState<string | null>(
    typeof window !== "undefined" ? localStorage.getItem("auth_token") : null,
  )

  const apiCall = useCallback(
    async (endpoint: string, options: RequestInit = {}): Promise<ApiResponse<any>> => {
      setLoading(true)
      try {
        const headers: HeadersInit = {
          "Content-Type": "application/json",
          ...options.headers,
        }

        if (token) {
          headers.Authorization = `Bearer ${token}`
        }

        const response = await fetch(`/api${endpoint}`, {
          ...options,
          headers,
        })

        const data = await response.json()

        return {
          data: response.ok ? data : undefined,
          error: response.ok ? undefined : data.detail || "An error occurred",
          status: response.status,
        }
      } catch (error) {
        return {
          error: "Network error occurred",
          status: 500,
        }
      } finally {
        setLoading(false)
      }
    },
    [token],
  )

  // Authentication endpoints
  const login = useCallback(
    async (credentials: LoginCredentials) => {
      const formData = new FormData()
      formData.append("username", credentials.username)
      formData.append("password", credentials.password)

      const result = await apiCall("/auth/login", {
        method: "POST",
        body: formData,
        headers: {}, // Remove Content-Type for FormData
      })

      if (result.data?.access_token) {
        setToken(result.data.access_token)
        localStorage.setItem("auth_token", result.data.access_token)
      }

      return result
    },
    [apiCall],
  )

  const signup = useCallback(
    async (data: SignupData) => {
      return apiCall("/auth/signup", {
        method: "POST",
        body: JSON.stringify(data),
      })
    },
    [apiCall],
  )

  // Consent management
  const recordConsent = useCallback(
    async (data: ConsentData) => {
      return apiCall("/consent/record", {
        method: "POST",
        body: JSON.stringify(data),
      })
    },
    [apiCall],
  )

  // Session management
  const startSession = useCallback(
    async (data: SessionData) => {
      return apiCall("/session/start", {
        method: "POST",
        body: JSON.stringify(data),
      })
    },
    [apiCall],
  )

  // ASR data ingestion
  const ingestSegments = useCallback(
    async (data: SegmentData) => {
      return apiCall("/asr/ingest", {
        method: "POST",
        body: JSON.stringify(data),
      })
    },
    [apiCall],
  )

  // Summary generation
  const generateSummary = useCallback(
    async (sessionId: number) => {
      return apiCall(`/summarize/${sessionId}`, {
        method: "POST",
      })
    },
    [apiCall],
  )

  // Review and approval
  const approveSummary = useCallback(
    async (summaryId: number) => {
      return apiCall(`/review/${summaryId}/approve`, {
        method: "POST",
      })
    },
    [apiCall],
  )

  // Patient Q&A
  const patientQuery = useCallback(
    async (patientId: number, query: string) => {
      return apiCall(`/patient/${patientId}/qa?q=${encodeURIComponent(query)}`)
    },
    [apiCall],
  )

  // Get summary
  const getSummary = useCallback(
    async (summaryId: number) => {
      return apiCall(`/summary/${summaryId}`)
    },
    [apiCall],
  )

  const logout = useCallback(() => {
    setToken(null)
    localStorage.removeItem("auth_token")
  }, [])

  return {
    loading,
    token,
    login,
    signup,
    logout,
    recordConsent,
    startSession,
    ingestSegments,
    generateSummary,
    approveSummary,
    patientQuery,
    getSummary,
  }
}
