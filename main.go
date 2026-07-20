// tf-anthropic-proxy: a small Anthropic Messages API -> OpenAI Chat Completions
// translation proxy for the tokenfactory gateway.
//
// Why it exists:
//   - The gateway's native /v1/messages streaming is non-compliant (it emits
//     OpenAI chat.completion.chunk objects instead of Anthropic SSE events), so
//     Anthropic clients (Claude Code, the Anthropic SDK with stream=true) break.
//   - The gateway also rejects the default Go/py User-Agent with a 403 (WAF).
//
// It speaks correct Anthropic Messages API (incl. real SSE and tool use) to the
// client and talks OpenAI /v1/chat/completions to the upstream, spoofing a curl
// User-Agent so the WAF lets it through.
//
// Supported: text chat, system prompts, and tool use (tools / tool_use /
// tool_result), streaming and non-streaming. Image blocks are not translated.
//
// Endpoints:
//   POST /v1/messages              (streaming + non-streaming)
//   POST /v1/messages/count_tokens (rough estimate)
//   GET  /v1/models                (proxied from upstream /models)
//   GET  /healthz
//
// Config (env):
//   UPSTREAM_BASE_URL  default https://api.tokenfactory.iamsaif.ai/v1
//   UPSTREAM_API_KEY   required (falls back to AIGATEWAY_API_KEY)
//   LISTEN_ADDR        default :4000
//   UPSTREAM_UA        default curl/8.4.0  (avoid the WAF 403)
//   DEBUG              if set, logs each incoming request body
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

var (
	upstreamBase = env("UPSTREAM_BASE_URL", "https://api.tokenfactory.iamsaif.ai/v1")
	upstreamKey  = firstNonEmpty(os.Getenv("UPSTREAM_API_KEY"), os.Getenv("AIGATEWAY_API_KEY"))
	listenAddr   = env("LISTEN_ADDR", ":4000")
	upstreamUA   = env("UPSTREAM_UA", "curl/8.4.0")
	debug        = os.Getenv("DEBUG") != ""
	httpClient   = &http.Client{Timeout: 10 * time.Minute}
)

// Claude Code's gateway model discovery only accepts model ids that start with
// "claude" or "anthropic". We alias each backend id with this prefix for the
// /v1/models listing, and strip it back off on incoming requests.
const modelAliasPrefix = "claude-proxy--"

func resolveModel(m string) string {
	return strings.TrimPrefix(m, modelAliasPrefix)
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// ---------- Anthropic request shapes ----------

type anthropicRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	Messages      []anthropicMessage `json:"messages"`
	System        json.RawMessage    `json:"system,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    json.RawMessage    `json:"tool_choice,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// contentBlock is a superset of the Anthropic block types we care about.
type contentBlock struct {
	Type string `json:"type"`
	// text
	Text string `json:"text"`
	// tool_use
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
	// tool_result
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

// parseBlocks accepts either a bare string or an array of content blocks and
// returns a normalized slice of blocks.
func parseBlocks(raw json.RawMessage) []contentBlock {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []contentBlock{{Type: "text", Text: s}}
	}
	var blocks []contentBlock
	if json.Unmarshal(raw, &blocks) == nil {
		return blocks
	}
	return nil
}

// blocksText concatenates the text of any text blocks (used for system and for
// tool_result content, which may itself be a string or an array of blocks).
func blocksText(raw json.RawMessage) string {
	var b strings.Builder
	for _, blk := range parseBlocks(raw) {
		if blk.Type == "text" {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// ---------- OpenAI shapes ----------

type openaiMessage struct {
	Role       string           `json:"role"`
	Content    *string          `json:"content"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openaiFunctionCall `json:"function"`
}

type openaiFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

type openaiTool struct {
	Type     string            `json:"type"`
	Function openaiFunctionDef `json:"function"`
}

type openaiFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type openaiRequest struct {
	Model         string          `json:"model"`
	Messages      []openaiMessage `json:"messages"`
	MaxTokens     int             `json:"max_tokens,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	Stop          []string        `json:"stop,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *streamOptions  `json:"stream_options,omitempty"`
	Tools         []openaiTool    `json:"tools,omitempty"`
	ToolChoice    any             `json:"tool_choice,omitempty"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type openaiResponse struct {
	Choices []struct {
		Message struct {
			Content   string           `json:"content"`
			ToolCalls []openaiToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage openaiUsage `json:"usage"`
}

type openaiChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *openaiUsage `json:"usage"`
}

// stopReason maps an OpenAI finish_reason to an Anthropic stop_reason.
func stopReason(fr string) string {
	switch fr {
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default: // "stop", "content_filter", ""
		return "end_turn"
	}
}

// ---------- request translation ----------

func strptr(s string) *string { return &s }

func convertToolChoice(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &tc) != nil {
		return nil
	}
	switch tc.Type {
	case "any":
		return "required"
	case "tool":
		return map[string]any{"type": "function", "function": map[string]string{"name": tc.Name}}
	case "auto", "none":
		return tc.Type
	default:
		return "auto"
	}
}

func toOpenAI(ar anthropicRequest, stream bool) openaiRequest {
	// Collect all system text (top-level `system` plus any stray role:"system"
	// messages) into a single leading system message; the upstream requires the
	// system message at index 0.
	var sysParts []string
	if s := blocksText(ar.System); s != "" {
		sysParts = append(sysParts, s)
	}

	var rest []openaiMessage
	for _, m := range ar.Messages {
		blocks := parseBlocks(m.Content)
		switch m.Role {
		case "system":
			if t := textOf(blocks); t != "" {
				sysParts = append(sysParts, t)
			}
		case "assistant":
			rest = append(rest, assistantToOpenAI(blocks))
		default: // user
			rest = append(rest, userToOpenAI(blocks)...)
		}
	}

	msgs := make([]openaiMessage, 0, len(rest)+1)
	if len(sysParts) > 0 {
		msgs = append(msgs, openaiMessage{Role: "system", Content: strptr(strings.Join(sysParts, "\n\n"))})
	}
	msgs = append(msgs, rest...)

	req := openaiRequest{
		Model:       ar.Model,
		Messages:    msgs,
		MaxTokens:   ar.MaxTokens,
		Temperature: ar.Temperature,
		TopP:        ar.TopP,
		Stop:        ar.StopSequences,
		Stream:      stream,
		ToolChoice:  convertToolChoice(ar.ToolChoice),
	}
	for _, t := range ar.Tools {
		req.Tools = append(req.Tools, openaiTool{
			Type:     "function",
			Function: openaiFunctionDef{Name: t.Name, Description: t.Description, Parameters: t.InputSchema},
		})
	}
	if stream {
		req.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return req
}

func textOf(blocks []contentBlock) string {
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type == "text" {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// assistantToOpenAI turns an assistant turn (text + tool_use blocks) into one
// OpenAI assistant message with optional tool_calls.
func assistantToOpenAI(blocks []contentBlock) openaiMessage {
	msg := openaiMessage{Role: "assistant"}
	if t := textOf(blocks); t != "" {
		msg.Content = strptr(t)
	}
	for _, blk := range blocks {
		if blk.Type == "tool_use" {
			args := string(blk.Input)
			if args == "" {
				args = "{}"
			}
			msg.ToolCalls = append(msg.ToolCalls, openaiToolCall{
				ID: blk.ID, Type: "function",
				Function: openaiFunctionCall{Name: blk.Name, Arguments: args},
			})
		}
	}
	return msg
}

// userToOpenAI turns a user turn into OpenAI messages. tool_result blocks each
// become a separate role:"tool" message; remaining text becomes a user message.
func userToOpenAI(blocks []contentBlock) []openaiMessage {
	var out []openaiMessage
	var textParts []string
	for _, blk := range blocks {
		switch blk.Type {
		case "tool_result":
			content := blocksText(blk.Content)
			if content == "" {
				// content may be a bare string already captured by blocksText;
				// fall back to the raw form if it wasn't a text/array-of-text.
				var s string
				if json.Unmarshal(blk.Content, &s) == nil {
					content = s
				}
			}
			if blk.IsError && content != "" {
				content = "[tool error] " + content
			}
			out = append(out, openaiMessage{
				Role: "tool", ToolCallID: blk.ToolUseID, Content: strptr(content),
			})
		case "text":
			textParts = append(textParts, blk.Text)
		}
	}
	if len(textParts) > 0 {
		out = append(out, openaiMessage{Role: "user", Content: strptr(strings.Join(textParts, "\n"))})
	}
	return out
}

func newUpstreamRequest(body []byte) (*http.Request, error) {
	req, err := http.NewRequest("POST", upstreamBase+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+upstreamKey)
	req.Header.Set("User-Agent", upstreamUA) // dodge the WAF 403
	return req, nil
}

// ---------- handlers ----------

func messagesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		anthropicError(w, http.StatusBadRequest, "invalid_request_error", "cannot read body")
		return
	}
	if debug {
		log.Printf("[req] %s", string(raw))
	}
	var ar anthropicRequest
	if err := json.Unmarshal(raw, &ar); err != nil {
		anthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error())
		return
	}
	ar.Model = resolveModel(ar.Model) // strip the discovery alias if present
	if ar.Model == "" || len(ar.Messages) == 0 {
		anthropicError(w, http.StatusBadRequest, "invalid_request_error", "model and messages are required")
		return
	}
	if ar.Stream {
		handleStream(w, ar)
	} else {
		handleNonStream(w, ar)
	}
}

func handleNonStream(w http.ResponseWriter, ar anthropicRequest) {
	body, _ := json.Marshal(toOpenAI(ar, false))
	req, err := newUpstreamRequest(body)
	if err != nil {
		anthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		anthropicError(w, http.StatusBadGateway, "api_error", "upstream: "+err.Error())
		return
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		anthropicError(w, resp.StatusCode, "api_error", "upstream "+resp.Status+": "+string(data))
		return
	}
	var or openaiResponse
	if err := json.Unmarshal(data, &or); err != nil || len(or.Choices) == 0 {
		anthropicError(w, http.StatusBadGateway, "api_error", "bad upstream response")
		return
	}
	ch := or.Choices[0]

	content := []map[string]any{}
	if ch.Message.Content != "" {
		content = append(content, map[string]any{"type": "text", "text": ch.Message.Content})
	}
	for _, tc := range ch.Message.ToolCalls {
		input := json.RawMessage(tc.Function.Arguments)
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		content = append(content, map[string]any{
			"type": "tool_use", "id": tc.ID, "name": tc.Function.Name, "input": input,
		})
	}
	if len(content) == 0 {
		content = append(content, map[string]any{"type": "text", "text": ""})
	}

	out := map[string]any{
		"id":            "msg_" + randID(),
		"type":          "message",
		"role":          "assistant",
		"model":         ar.Model,
		"content":       content,
		"stop_reason":   stopReason(ch.FinishReason),
		"stop_sequence": nil,
		"usage": map[string]int{
			"input_tokens":  or.Usage.PromptTokens,
			"output_tokens": or.Usage.CompletionTokens,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// ---------- streaming ----------

// sseState tracks the single currently-open Anthropic content block while we
// translate the OpenAI chunk stream. Anthropic requires each content block to be
// explicitly started and stopped, one at a time, with incrementing indices.
type sseState struct {
	w        io.Writer
	f        http.Flusher
	nextIdx  int    // next Anthropic content-block index to assign
	openKind string // "", "text", or "tool"
	openIdx  int    // Anthropic index of the currently-open block
	toolOA   int    // OpenAI tool index currently open (-1 if none)
	sawTool  bool
}

func (s *sseState) closeOpen() {
	if s.openKind == "" {
		return
	}
	writeSSE(s.w, s.f, "content_block_stop", map[string]any{"type": "content_block_stop", "index": s.openIdx})
	s.openKind = ""
}

func (s *sseState) textDelta(text string) {
	if s.openKind != "text" {
		s.closeOpen()
		s.openIdx = s.nextIdx
		s.nextIdx++
		s.openKind = "text"
		writeSSE(s.w, s.f, "content_block_start", map[string]any{
			"type": "content_block_start", "index": s.openIdx,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
	}
	writeSSE(s.w, s.f, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": s.openIdx,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
}

func (s *sseState) startTool(oaIdx int, id, name string) {
	s.closeOpen()
	s.openIdx = s.nextIdx
	s.nextIdx++
	s.openKind = "tool"
	s.toolOA = oaIdx
	s.sawTool = true
	writeSSE(s.w, s.f, "content_block_start", map[string]any{
		"type": "content_block_start", "index": s.openIdx,
		"content_block": map[string]any{"type": "tool_use", "id": id, "name": name, "input": map[string]any{}},
	})
}

func (s *sseState) toolArgsDelta(text string) {
	if s.openKind != "tool" {
		return
	}
	writeSSE(s.w, s.f, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": s.openIdx,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": text},
	})
}

func handleStream(w http.ResponseWriter, ar anthropicRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		anthropicError(w, http.StatusInternalServerError, "api_error", "streaming unsupported")
		return
	}
	body, _ := json.Marshal(toOpenAI(ar, true))
	req, err := newUpstreamRequest(body)
	if err != nil {
		anthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		anthropicError(w, http.StatusBadGateway, "api_error", "upstream: "+err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		anthropicError(w, resp.StatusCode, "api_error", "upstream "+resp.Status+": "+string(data))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	msgID := "msg_" + randID()
	writeSSE(w, flusher, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": msgID, "type": "message", "role": "assistant", "model": ar.Model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]int{"input_tokens": 0, "output_tokens": 0},
		},
	})
	writeSSE(w, flusher, "ping", map[string]any{"type": "ping"})

	st := &sseState{w: w, f: flusher, toolOA: -1}
	finish := "stop"
	usage := openaiUsage{}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(line[5:])
		if payload == "[DONE]" {
			break
		}
		var chunk openaiChunk
		if json.Unmarshal([]byte(payload), &chunk) != nil {
			continue
		}
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		d := chunk.Choices[0]
		if d.Delta.Content != "" {
			st.textDelta(d.Delta.Content)
		}
		for _, tc := range d.Delta.ToolCalls {
			// A tool block begins when the upstream sends an id or a name for it.
			if tc.ID != "" || tc.Function.Name != "" {
				st.startTool(tc.Index, tc.ID, tc.Function.Name)
			}
			if tc.Function.Arguments != "" {
				st.toolArgsDelta(tc.Function.Arguments)
			}
		}
		if d.FinishReason != nil && *d.FinishReason != "" {
			finish = *d.FinishReason
		}
	}

	// If the model produced nothing, emit an empty text block so the message has
	// at least one content block.
	if st.nextIdx == 0 {
		st.textDelta("")
	}
	st.closeOpen()

	sr := stopReason(finish)
	if st.sawTool {
		sr = "tool_use"
	}
	writeSSE(w, flusher, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": sr, "stop_sequence": nil},
		"usage": map[string]int{"output_tokens": usage.CompletionTokens},
	})
	writeSSE(w, flusher, "message_stop", map[string]any{"type": "message_stop"})
}

type upstreamModelList struct {
	Data []struct {
		ID      string `json:"id"`
		Created int64  `json:"created"`
		Mode    string `json:"mode"`
	} `json:"data"`
}

// modelsHandler fetches the upstream (OpenAI-shaped) model list and returns it
// in Anthropic's /v1/models schema, aliasing each id with modelAliasPrefix so
// Claude Code's gateway discovery (which requires a claude*/anthropic* id) will
// list them. Only chat-capable models are included.
func modelsHandler(w http.ResponseWriter, r *http.Request) {
	req, err := http.NewRequest("GET", upstreamBase+"/models", nil)
	if err != nil {
		anthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	req.Header.Set("Authorization", "Bearer "+upstreamKey)
	req.Header.Set("User-Agent", upstreamUA)
	resp, err := httpClient.Do(req)
	if err != nil {
		anthropicError(w, http.StatusBadGateway, "api_error", "upstream: "+err.Error())
		return
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		anthropicError(w, resp.StatusCode, "api_error", "upstream "+resp.Status+": "+string(data))
		return
	}
	var ul upstreamModelList
	if json.Unmarshal(data, &ul) != nil {
		anthropicError(w, http.StatusBadGateway, "api_error", "bad upstream model list")
		return
	}
	items := make([]map[string]any, 0, len(ul.Data))
	for _, m := range ul.Data {
		if m.Mode != "" && m.Mode != "chat" {
			continue // skip embedding / audio / tts models
		}
		items = append(items, map[string]any{
			"type":         "model",
			"id":           modelAliasPrefix + m.ID,
			"display_name": m.ID,
			"created_at":   time.Unix(m.Created, 0).UTC().Format(time.RFC3339),
		})
	}
	out := map[string]any{"data": items, "has_more": false, "first_id": nil, "last_id": nil}
	if len(items) > 0 {
		out["first_id"] = items[0]["id"]
		out["last_id"] = items[len(items)-1]["id"]
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// modelRetrieveHandler answers GET /v1/models/{id} in Anthropic's schema.
func modelRetrieveHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/models/")
	real := resolveModel(id)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"type":         "model",
		"id":           modelAliasPrefix + real,
		"display_name": real,
		"created_at":   time.Unix(0, 0).UTC().Format(time.RFC3339),
	})
}

// count_tokens: the gateway doesn't expose one, so return a rough estimate
// (~4 chars/token). Good enough for clients that only need a ballpark.
func countTokensHandler(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var ar anthropicRequest
	json.Unmarshal(raw, &ar)
	chars := len(blocksText(ar.System))
	for _, m := range ar.Messages {
		for _, blk := range parseBlocks(m.Content) {
			chars += len(blk.Text) + len(blk.Input) + len(blk.Content)
		}
	}
	est := chars / 4
	if est < 1 && chars > 0 {
		est = 1
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int{"input_tokens": est})
}

// ---------- helpers ----------

func writeSSE(w io.Writer, f http.Flusher, event string, data any) {
	b, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	f.Flush()
}

func anthropicError(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{
		"type":  "error",
		"error": map[string]string{"type": typ, "message": msg},
	})
}

// randID returns a short, time-derived hex id (no crypto needed here).
func randID() string {
	return fmt.Sprintf("%013x", time.Now().UnixNano())
}

func main() {
	if upstreamKey == "" {
		log.Fatal("set UPSTREAM_API_KEY (or AIGATEWAY_API_KEY) to your tokenfactory key")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", messagesHandler)
	mux.HandleFunc("/v1/messages/count_tokens", countTokensHandler)
	mux.HandleFunc("/v1/models", modelsHandler)
	mux.HandleFunc("/v1/models/", modelRetrieveHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})
	log.Printf("tf-anthropic-proxy listening on %s -> %s", listenAddr, upstreamBase)
	log.Fatal(http.ListenAndServe(listenAddr, mux))
}
