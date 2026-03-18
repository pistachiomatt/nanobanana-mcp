# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What This Is

NanoBanana MCP — a Model Context Protocol server exposing Google Gemini's image generation, editing, and vision/chat capabilities over stdio. Single-file server at `src/index.ts`.

## Build & Run

```bash
npm run build        # tsc → dist/
npm run dev          # tsx watch (hot reload)
npm run start        # node dist/index.js
```

**dist/ is committed.** After any change to `src/index.ts`, always `npm run build` so `dist/index.js` stays in sync. This enables direct GitHub-based `npx` usage without a build step.

## Architecture

Everything lives in `src/index.ts`. No multi-file abstractions — it's intentionally a single file. The structure:

1. **CLI handler** (lines ~7–47) — `--install-commands <claude-code|cursor>` copies slash commands
2. **Gemini API layer** — `callGeminiImageAPI()` does raw REST calls with streaming response parsing
3. **Session management** — `conversations` Map keyed by `conversation_id`, holding chat history, image history, aspect ratio, and model selection
4. **MCP tools** — registered via `ListToolsRequestSchema` / `CallToolRequestSchema`

### Tools

| Tool | Purpose |
|------|---------|
| `gemini_chat` | Multi-turn conversation with optional images |
| `gemini_generate_image` | Image generation with consistency/reference support |
| `gemini_edit_image` | Natural language image editing |
| `set_aspect_ratio` | Set session aspect ratio (required before image gen/edit) |
| `set_model` | Switch flash/pro per session |
| `get_image_history` | List generated/edited images in session |
| `clear_conversation` | Reset session context |

### Prompt Resolution

All three content tools (`gemini_chat`, `gemini_generate_image`, `gemini_edit_image`) support **file-based prompts** as an alternative to inline strings:

- `message` / `message_file`
- `prompt` / `prompt_file`
- `edit_prompt` / `edit_prompt_file`

The `resolvePrompt()` helper reads the file and returns trimmed contents. File param takes priority if both are provided.

### Image References

Session images can be referenced by `"last"` or `"history:N"` (e.g., `"history:0"`). This works in `image_path`, `images`, and `reference_images` params. Resolved via `getImageFromHistory()`.

### File Path Resolution

Image/file paths resolve in order: absolute → relative to cwd → fallback to `~/Documents/nanobanana_generated/`.

## Configuration

| Env var | Required | Purpose |
|---------|----------|---------|
| `GOOGLE_AI_API_KEY` | Yes | Gemini API key |
| `NANOBANANA_MODEL` | No | Default model (`gemini-3.1-flash-image-preview`) |
| `NANOBANANA_PATH_ONLY` | No | `"true"` to skip inline base64 in responses |

## Patterns to Follow

- **Keep it single-file.** Resist the urge to split. The whole server is ~1100 lines — that's fine.
- **Tool params are loose.** Destructured as `args as any`. No runtime validation library. Keep it simple.
- **Korean comments are from the original author.** Leave them; they're navigational landmarks.
- **Session defaults.** `conversation_id` defaults to `"default"` in every tool handler. Aspect ratio has no default — must be set explicitly.
- **Error responses** use `isError: true` in the MCP content response, not thrown exceptions (those are caught by the outer try/catch).

## Slash Commands

`commands/claude-code/` and `commands/cursor/` contain IDE-specific slash commands (`nb-flash.md`, `nb-pro.md`). Install with `--install-commands` CLI flag or copy manually.

## Adding a New Tool

1. Add schema to the `tools` array in `ListToolsRequestSchema` handler
2. Add case to the `switch` in `CallToolRequestSchema` handler
3. If it needs text input, use `resolvePrompt()` with both inline and `_file` params
4. Rebuild: `npm run build`
