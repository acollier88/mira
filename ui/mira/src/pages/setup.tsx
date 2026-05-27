import { Loader2 } from "lucide-react"
import { useEffect, useState } from "react"
import { useNavigate } from "react-router"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { api } from "@/lib/api"

type ModelOption = {
  value: string
  label: string
  recommended?: boolean
}

export function SetupPage() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [indexingModel, setIndexingModel] = useState("")
  const [reviewModel, setReviewModel] = useState("")
  const [baseUrl, setBaseUrl] = useState("")
  const [apiKeyEnv, setApiKeyEnv] = useState("OPENROUTER_API_KEY")
  const [authMode, setAuthMode] = useState<"bearer" | "none">("bearer")
  const [indexingOptions, setIndexingOptions] = useState<ModelOption[]>([])
  const [reviewOptions, setReviewOptions] = useState<ModelOption[]>([])
  const [error, setError] = useState("")

  useEffect(() => {
    api.getModels().then((data) => {
      setIndexingModel(data.indexing_model)
      setReviewModel(data.review_model)
      setBaseUrl(data.base_url)
      setApiKeyEnv(data.api_key_env || "OPENROUTER_API_KEY")
      setAuthMode(data.api_key_env === "" ? "none" : "bearer")
      setIndexingOptions(data.indexing_options)
      setReviewOptions(data.review_options)
      setLoading(false)
    })
  }, [])

  const handleSave = async () => {
    setSaving(true)
    setError("")
    try {
      await api.saveModels({
        indexing_model: indexingModel,
        review_model: reviewModel,
        base_url: baseUrl,
        api_key_env: authMode === "none" ? "" : apiKeyEnv,
      })
      navigate("/")
    } catch (err) {
      const raw = err instanceof Error ? err.message : String(err)
      let parsedError: { detail?: { field?: string; message: string } } | null = null
      try { parsedError = JSON.parse(raw.replace(/^API error \d+: /, "")) } catch { /* ignore */ }
      const detail = parsedError?.detail
      setError(
        detail && typeof detail === "object" && "message" in detail
          ? `${detail.field ? `${detail.field}: ` : ""}${detail.message}`
          : raw,
      )
    } finally {
      setSaving(false)
    }
  }

  const knownModelValue = (value: string, options: ModelOption[]) =>
    options.some((opt) => opt.value === value) ? value : undefined

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-lg space-y-6 px-4 py-16">
      <div className="text-center">
        <img src="/logo.png" alt="Mira" className="mx-auto mb-4 h-12 w-12" />
        <h1 className="text-2xl font-semibold tracking-tight">
          Welcome to Mira
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Choose models and endpoint settings for indexing and reviews
        </p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Indexing Model</CardTitle>
          <CardDescription>
            Used to summarize files when building the code index. We recommend
            a cheaper model here since it runs over every file.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Select
            value={knownModelValue(indexingModel, indexingOptions)}
            onValueChange={setIndexingModel}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a known indexing model" />
            </SelectTrigger>
            <SelectContent>
              {indexingOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                  {opt.recommended && (
                    <span className="ml-2 text-xs text-muted-foreground">
                      Recommended
                    </span>
                  )}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            className="mt-3"
            value={indexingModel}
            onChange={(e) => setIndexingModel(e.target.value)}
            placeholder="Custom indexing model ID"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Review Model</CardTitle>
          <CardDescription>
            Used to analyze PRs and post comments. A more powerful model here
            gives better review quality.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Select
            value={knownModelValue(reviewModel, reviewOptions)}
            onValueChange={setReviewModel}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a known review model" />
            </SelectTrigger>
            <SelectContent>
              {reviewOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                  {opt.recommended && (
                    <span className="ml-2 text-xs text-muted-foreground">
                      Recommended
                    </span>
                  )}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            className="mt-3"
            value={reviewModel}
            onChange={(e) => setReviewModel(e.target.value)}
            placeholder="Custom review model ID"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">LLM Endpoint</CardTitle>
          <CardDescription>
            Point Mira at OpenRouter or any OpenAI-compatible endpoint such as Ollama, SGLang, or vLLM.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2">
            <label className="text-sm font-medium">Base URL</label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://openrouter.ai/api/v1"
            />
            <p className="text-xs text-muted-foreground">
              Mira appends <code className="text-xs">/chat/completions</code> to this URL.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Authentication</label>
            <Select
              value={authMode}
              onValueChange={(value) => setAuthMode(value as "bearer" | "none")}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bearer">****** from environment</SelectItem>
                <SelectItem value="none">No auth</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {authMode === "bearer" && (
            <div className="space-y-2">
              <label className="text-sm font-medium">API key env var</label>
              <Input
                value={apiKeyEnv}
                onChange={(e) => setApiKeyEnv(e.target.value)}
                placeholder="OPENROUTER_API_KEY"
              />
              <p className="text-xs text-muted-foreground">
                Mira reads the bearer token from this environment variable.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <p className="text-center text-xs text-muted-foreground">
        You can change these later in Settings
      </p>

      {error && (
        <p className="text-center text-xs text-destructive break-words">
          {error}
        </p>
      )}

      <Button className="w-full" size="lg" onClick={handleSave} disabled={saving}>
        {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        Save and Continue
      </Button>
    </div>
  )
}
