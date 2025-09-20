"use client"

import { Label } from "@/components/ui/label"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Shield, FileText, Lock, Eye, CheckCircle } from "lucide-react"
import { useRouter } from "next/navigation"

export default function ConsentPage() {
  const [hasReadConsent, setHasReadConsent] = useState(false)
  const [agreedToConsent, setAgreedToConsent] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const router = useRouter()

  const handleSubmit = async () => {
    if (!hasReadConsent || !agreedToConsent) return

    setIsSubmitting(true)
    // TODO: Submit consent to backend API
    setTimeout(() => {
      setIsSubmitting(false)
      router.push("/auth/signup")
    }, 2000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-medical-warm to-background p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-lg bg-primary flex items-center justify-center">
              <Shield className="w-7 h-7 text-primary-foreground" />
            </div>
            <h1 className="text-3xl font-bold text-foreground">Informed Consent</h1>
          </div>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Before we begin, please review how we protect your privacy and use your medical information.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Privacy Features */}
          <div className="lg:col-span-1 space-y-4">
            <Card className="warm-card border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lock className="w-5 h-5 text-primary" />
                  Your Privacy
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-success mt-0.5" />
                  <div>
                    <p className="font-medium text-sm">Personal Information Protected</p>
                    <p className="text-xs text-muted-foreground">
                      Names, dates, and IDs are automatically hidden from AI
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-success mt-0.5" />
                  <div>
                    <p className="font-medium text-sm">Secure Data Storage</p>
                    <p className="text-xs text-muted-foreground">All information encrypted and stored securely</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-success mt-0.5" />
                  <div>
                    <p className="font-medium text-sm">You Control Access</p>
                    <p className="text-xs text-muted-foreground">
                      Only you and your healthcare provider can see your data
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="warm-card border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="w-5 h-5 text-primary" />
                  What You'll Get
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-success mt-0.5" />
                  <div>
                    <p className="font-medium text-sm">Easy-to-Understand Summaries</p>
                    <p className="text-xs text-muted-foreground">Clear explanations of your medical visits</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-success mt-0.5" />
                  <div>
                    <p className="font-medium text-sm">Ask Questions Anytime</p>
                    <p className="text-xs text-muted-foreground">Search your medical history and get answers</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-success mt-0.5" />
                  <div>
                    <p className="font-medium text-sm">Track Your Care</p>
                    <p className="text-xs text-muted-foreground">See how your health journey progresses over time</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Consent Document */}
          <div className="lg:col-span-2">
            <Card className="warm-card border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-primary" />
                  Informed Consent for Medical AI Services
                </CardTitle>
                <CardDescription>
                  Please read this information carefully before agreeing to use our services.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-96 w-full rounded-md border border-border/50 p-4 bg-background/50">
                  <div className="space-y-4 text-sm">
                    <section>
                      <h3 className="font-semibold mb-2">1. Purpose of This Service</h3>
                      <p className="text-muted-foreground">
                        Nightingale AI helps create easy-to-understand summaries of your medical visits. Our AI listens
                        to your conversations with healthcare providers and creates two types of summaries: one for your
                        doctor's records and one written in simple language for you.
                      </p>
                    </section>

                    <section>
                      <h3 className="font-semibold mb-2">2. How We Protect Your Privacy</h3>
                      <p className="text-muted-foreground mb-2">
                        Your privacy is our top priority. Here's how we protect your information:
                      </p>
                      <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
                        <li>
                          Personal details (names, dates, ID numbers) are automatically hidden before any AI processing
                        </li>
                        <li>All data is encrypted during transmission and storage</li>
                        <li>Only you and your healthcare provider can access your information</li>
                        <li>We never share your data with third parties without your explicit permission</li>
                        <li>You can request deletion of your data at any time</li>
                      </ul>
                    </section>

                    <section>
                      <h3 className="font-semibold mb-2">3. What Information We Collect</h3>
                      <p className="text-muted-foreground mb-2">
                        We only collect information necessary to provide our services:
                      </p>
                      <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
                        <li>Audio recordings of your medical consultations (with your permission)</li>
                        <li>Text transcriptions of these conversations (automatically privacy-protected)</li>
                        <li>Basic account information (username, encrypted password)</li>
                        <li>Your feedback and edits to summaries</li>
                      </ul>
                    </section>

                    <section>
                      <h3 className="font-semibold mb-2">4. Your Rights and Control</h3>
                      <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
                        <li>You can stop recording at any time during a consultation</li>
                        <li>You can edit or request changes to any summary</li>
                        <li>You can withdraw consent and delete your data at any time</li>
                        <li>You can request a copy of all your data</li>
                        <li>You control who can access your information</li>
                      </ul>
                    </section>

                    <section>
                      <h3 className="font-semibold mb-2">5. Data Security</h3>
                      <p className="text-muted-foreground">
                        We use industry-standard security measures including encryption, secure servers, and regular
                        security audits. All staff accessing your data are trained in privacy protection and bound by
                        confidentiality agreements.
                      </p>
                    </section>

                    <section>
                      <h3 className="font-semibold mb-2">6. Contact Information</h3>
                      <p className="text-muted-foreground">
                        If you have questions about this consent or our privacy practices, please contact our Privacy
                        Officer at privacy@nightingale-ai.com or through your healthcare provider.
                      </p>
                    </section>
                  </div>
                </ScrollArea>

                {/* Consent Checkboxes */}
                <div className="mt-6 space-y-4">
                  <div className="flex items-start space-x-3">
                    <Checkbox
                      id="read-consent"
                      checked={hasReadConsent}
                      onCheckedChange={(checked) => setHasReadConsent(checked as boolean)}
                    />
                    <Label htmlFor="read-consent" className="text-sm leading-relaxed">
                      I have read and understand the information above about how my medical information will be used and
                      protected.
                    </Label>
                  </div>

                  <div className="flex items-start space-x-3">
                    <Checkbox
                      id="agree-consent"
                      checked={agreedToConsent}
                      onCheckedChange={(checked) => setAgreedToConsent(checked as boolean)}
                      disabled={!hasReadConsent}
                    />
                    <Label htmlFor="agree-consent" className="text-sm leading-relaxed">
                      I agree to use Nightingale AI services and consent to the collection, use, and protection of my
                      medical information as described above.
                    </Label>
                  </div>
                </div>

                {/* Submit Button */}
                <div className="mt-8 flex gap-4">
                  <Button
                    onClick={handleSubmit}
                    disabled={!hasReadConsent || !agreedToConsent || isSubmitting}
                    className="flex-1"
                    size="lg"
                  >
                    {isSubmitting ? "Processing..." : "I Agree - Continue to Dashboard"}
                  </Button>
                  <Button variant="outline" size="lg" asChild>
                    <a href="/">Cancel</a>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
