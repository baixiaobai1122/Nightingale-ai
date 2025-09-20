import { Shield, Stethoscope, Users, FileText, Lock, Zap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm bg-background/80">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Stethoscope className="w-5 h-5 text-primary-foreground" />
            </div>
            <h1 className="text-xl font-semibold text-foreground">Nightingale AI</h1>
          </div>
          <nav className="flex items-center gap-6">
            <Link href="/auth/login" className="text-muted-foreground hover:text-foreground transition-colors">
              Login In
            </Link>
            <Button asChild>
              <Link href="/auth/signup">Get Started</Link>
            </Button>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4 bg-gradient-to-b from-medical-warm to-background">
        <div className="container mx-auto text-center max-w-4xl">
          <div className="mb-6">
            <span className="privacy-badge text-sm font-medium px-3 py-1 rounded-full border border-primary/20 bg-primary/10">
              Your privacy is our priority
            </span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold text-balance mb-6">
            Medical AI that keeps your <span className="privacy-badge">information safe</span>
          </h1>
          <p className="text-xl text-muted-foreground text-balance mb-8 max-w-2xl mx-auto">
            Secure voice AI for medical consultations with automatic note-taking, privacy-protected summaries, and
            complete conversation tracking.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/auth/role-select">Start Free Trial</Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link href="/demo">View Demo</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Built for healthcare professionals and patients</h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Every feature designed with your privacy, conversation tracking, and healthcare workflow in mind.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="warm-card border-border/50">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
                  <Shield className="w-6 h-6 text-primary" />
                </div>
                <CardTitle>Privacy Protection</CardTitle>
                <CardDescription>
                  Your personal information is automatically protected before any AI processing.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Personal details are hidden from AI</li>
                  <li>• Secure data transmission</li>
                  <li>• Complete access control</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-success/20 flex items-center justify-center mb-4">
                  <FileText className="w-6 h-6 text-success" />
                </div>
                <CardTitle>Two Types of Summaries</CardTitle>
                <CardDescription>
                  Get both a detailed medical summary and a patient-friendly version from the same visit.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Detailed notes for doctors</li>
                  <li>• Easy-to-understand patient summaries</li>
                  <li>• Links back to original conversation</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-warning/20 flex items-center justify-center mb-4">
                  <Zap className="w-6 h-6 text-warning" />
                </div>
                <CardTitle>Fast Response</CardTitle>
                <CardDescription>
                  Real-time conversation processing optimized for busy medical environments.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Quick response times</li>
                  <li>• Optimized for clinic use</li>
                  <li>• Performance monitoring</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
                  <Lock className="w-6 h-6 text-primary" />
                </div>
                <CardTitle>Permission Management</CardTitle>
                <CardDescription>
                  Nothing happens without your explicit permission - you're always in control.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Clear permission requests</li>
                  <li>• Detailed access controls</li>
                  <li>• Complete activity logs</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-success/20 flex items-center justify-center mb-4">
                  <Users className="w-6 h-6 text-success" />
                </div>
                <CardTitle>Human Review</CardTitle>
                <CardDescription>
                  Doctors and patients can review, edit, and approve all summaries before they're final.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Review before finalizing</li>
                  <li>• Edit and approval system</li>
                  <li>• Version history tracking</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-warning/20 flex items-center justify-center mb-4">
                  <FileText className="w-6 h-6 text-warning" />
                </div>
                <CardTitle>Complete Conversation Tracking</CardTitle>
                <CardDescription>
                  Every summary point links back to the exact moment in your conversation.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Links to original conversation</li>
                  <li>• Time-stamped references</li>
                  <li>• Verified accuracy</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section className="py-20 px-4 border-t border-border/50">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Complete care workflow</h2>
            <p className="text-muted-foreground text-lg">
              From pre-care to post-care, every step is privacy-protected and auditable.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-6">
                <span className="text-2xl font-bold text-primary">1</span>
              </div>
              <h3 className="text-xl font-semibold mb-4">Pre-Care</h3>
              <p className="text-muted-foreground">
                Patient authentication, consent logging, and concern surfacing with clinician dossier preparation.
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-success/20 flex items-center justify-center mx-auto mb-6">
                <span className="text-2xl font-bold text-success">2</span>
              </div>
              <h3 className="text-xl font-semibold mb-4">During Care</h3>
              <p className="text-muted-foreground">
                Real-time recording, transcription, and generation of provenance-grounded summaries for both clinician
                and patient.
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-warning/20 flex items-center justify-center mx-auto mb-6">
                <span className="text-2xl font-bold text-warning">3</span>
              </div>
              <h3 className="text-xl font-semibold mb-4">Post-Care</h3>
              <p className="text-muted-foreground">
                Patient query system with persistent, context-aware memory for future consultations.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 border-t border-border/50 bg-medical-warm">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to improve your healthcare experience?</h2>
          <p className="text-muted-foreground text-lg mb-8 max-w-2xl mx-auto">
            Join healthcare professionals and patients who trust Nightingale AI for secure, privacy-first medical
            consultations.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/auth/role-select">Start Free Trial</Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link href="/contact">Contact Us</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 py-12 px-4">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-3 mb-4 md:mb-0">
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                <Stethoscope className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-semibold">Nightingale AI</span>
            </div>
            <div className="flex gap-6 text-sm text-muted-foreground">
              <Link href="/privacy" className="hover:text-foreground transition-colors">
                Privacy
              </Link>
              <Link href="/terms" className="hover:text-foreground transition-colors">
                Terms
              </Link>
              <Link href="/security" className="hover:text-foreground transition-colors">
                Security
              </Link>
              <Link href="/docs" className="hover:text-foreground transition-colors">
                Documentation
              </Link>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-border/50 text-center text-sm text-muted-foreground">
            <p>© 2024 Nightingale AI. Built with privacy, security, and provenance at its core.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
