# Document Search with Cloudflare Vectorize and OpenAI

This project provides a semantic document search system that ingests documents from the `dummy-data/` directory, generates OpenAI embeddings, and stores them in a Cloudflare Vectorize index. A Cloudflare Worker exposes both a semantic search API and a streaming question-answering endpoint that powers the included web UI.

## Components

- `embed_files.py` – Python utility that processes documents, generates embeddings, and upserts them into Cloudflare Vectorize
- `embedding-worker.js` – Cloudflare Worker script that queries the vectorize index and provides search/answer endpoints
- `index.html` – Web interface for searching documents and displaying results with AI-powered summaries
- `wrangler.toml` – Worker configuration with Vectorize binding and deployment settings

## Prerequisites

1. **Cloudflare account with Vectorize access** – create an index (the scripts expect `docsearch`)
2. **Cloudflare API token** – must include `Vectorize Read` and `Vectorize Write` permissions
3. **Cloudflare account ID** – available in the Cloudflare dashboard
4. **OpenAI API key** – used for both embeddings and GPT responses
5. **Python 3.8+** and optionally [uv](https://github.com/astral-sh/uv) for dependency management

## Environment configuration

Create a `.env` file (or export the variables) with the following values:

```env
OPENAI_API_KEY=sk-...
CLOUDFLARE_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CLOUDFLARE_API_TOKEN=cf_api_token_with_vectorize_access
CLOUDFLARE_VECTORIZE_INDEX=docsearch
```

The embedding script prompts interactively if any values are missing.

## Ingest documents into Cloudflare Vectorize

Run the embedding script from the repository root. The script ensures that the `docsearch` index exists, generates OpenAI embeddings for each file in `dummy-data/`, and upserts metadata and vectors into Cloudflare Vectorize.

```bash
uv run embed_files.py
# or: python embed_files.py
```

Each vector stores the filename, path, file size, extension, SHA256 content hash, and full document content as metadata. Empty files are skipped.

## Cloudflare Worker deployment

The `wrangler.toml` configuration file binds the `docsearch` Vectorize index:

```toml
[[vectorize]]
binding = "DOCSEARCH_VECTORIZE"
index_name = "docsearch"
```

Deploy the worker using:

```bash
wrangler deploy
```

Ensure your secrets are configured before deploying:

```bash
wrangler secret put OPENAI_API_KEY
```

During request handling the Worker:

1. Generates an embedding for the incoming query (OpenAI `text-embedding-3-small`)
2. Queries the bound Vectorize index (`DOCSEARCH_VECTORIZE`) for the most relevant documents
3. Returns JSON results for `/search`
4. Streams GPT-4.1-mini responses with supporting document chunks for `/answer`

## Local development

1. Create a `.env` file with your credentials:

   ```bash
   # Create .env file with your OpenAI and Cloudflare credentials
   OPENAI_API_KEY=sk-...
   CLOUDFLARE_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   CLOUDFLARE_API_TOKEN=cf_api_token_with_vectorize_access
   CLOUDFLARE_VECTORIZE_INDEX=docsearch
   ```

2. Populate Cloudflare Vectorize by running the embedding script (see previous section).

3. Start a local web server to serve the UI:

   ```bash
   python -m http.server 8000
   # or: npx serve
   ```

4. Open `http://localhost:8000/index.html` in your browser.

5. Update the `WORKER_URL` constant in `index.html` (line 134) to point to your deployed Worker:

   ```javascript
   const WORKER_URL = 'https://your-worker-name.workers.dev';
   ```

6. Test the search functionality and summarization features.

## Test embeddings locally

1. Ensure `.env` contains the credentials listed above. Optionally set `DOCSEARCH_DATA_DIR` if your documents live outside `dummy-data/`.
2. Run the embedding script:

   ```bash
   uv run embed_files.py
   # or: python embed_files.py
   ```

   The script confirms the index exists, reads every file in the configured directory, generates OpenAI embeddings, and upserts them into Cloudflare Vectorize with metadata.
3. Inspect the log output for success and failure counts. Re-run if any items failed (only changed files are reprocessed).
4. Verify the vectors by querying your Worker’s `/search` endpoint (replace the URL with your deployment or `wrangler dev` endpoint):

   ```bash
   curl -s -X POST "$DOCSEARCH_WORKER_URL/search" \
     -H "Content-Type: application/json" \
     -d '{"q":"sample question","ndocs":3}' | jq
   ```

## Web search interface

The `index.html` file provides a modern web interface for document search. The interface features:

- Semantic search with real-time results
- AI-powered document summarization
- Follow-up question suggestions
- Responsive design with dark theme
- Document preview and expansion capabilities

Configure the worker URL by updating the `WORKER_URL` constant in the JavaScript section of `index.html`. All searches are proxied through the `/search` endpoint, ensuring credentials never leave the Worker environment.

## Cloudflare R2 and secrets setup

If you plan to back up raw documents or store additional assets in Cloudflare R2, follow these steps to provision a bucket and capture the credentials needed for automation or server-side ingestion scripts:

1. **Create the R2 bucket**  
   - From the Cloudflare dashboard, navigate to **R2** → **Create bucket**.  
   - Choose a unique bucket name (for example `docsearch-ingest`) and confirm the region.

2. **Generate an R2 API token**  
   - Go to **My Profile** → **API Tokens** → **Create Token** → select the **R2 Edit** template.  
   - Scope the token to your account and the specific bucket you just created, granting **Read**, **Write**, and **Delete** permissions as required.  
   - Copy the token value (this becomes `CLOUDFLARE_R2_TOKEN`) and note the **Account ID** (already used elsewhere as `CLOUDFLARE_ACCOUNT_ID`).

3. **Save the access keys**  
   - After creating the token, Cloudflare shows an **Access Key ID** and **Secret Access Key**.  
   - Store these securely; they map to `CLOUDFLARE_R2_ACCESS_KEY_ID` and `CLOUDFLARE_R2_SECRET_ACCESS_KEY` for scripts or services that upload to R2.

4. **Bind the bucket in Wrangler (optional)**  
   - If the Worker needs direct R2 access, add a binding in `wrangler.toml`:
     ```toml
     [[r2_buckets]]
     binding = "DOCSEARCH_R2"
     bucket_name = "docsearch-ingest"
     ```
   - Deploy the Worker and use `env.DOCSEARCH_R2` inside the Worker to read/write R2 objects.

5. **Add secrets to the Worker**  
   - Run the following commands to store sensitive values:
     ```bash
     wrangler secret put CLOUDFLARE_R2_TOKEN
     wrangler secret put CLOUDFLARE_R2_ACCESS_KEY_ID
     wrangler secret put CLOUDFLARE_R2_SECRET_ACCESS_KEY
     ```
   - These secrets are available via `env.CLOUDFLARE_R2_TOKEN`, etc., inside the Worker.

6. **Update local environment**  
   - Mirror the same variables in your `.env` so local scripts (like `embed_files.py`) can upload to the bucket when needed.

## Verifying the index

Use the Worker’s `/search` endpoint to confirm that documents are available:

```bash
curl -s -X POST "https://<your-worker>.workers.dev/search" \
  -H "Content-Type: application/json" \
  -d '{"q": "admission requirements", "ndocs": 5}' | jq
```

## Environment variables summary

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI key for embeddings and chat completion |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID used for REST fallback requests |
| `CLOUDFLARE_API_TOKEN` | Token with Vectorize read/write permissions |
| `CLOUDFLARE_VECTORIZE_INDEX` | Vectorize index name (defaults to `docsearch`) |
| `DOCSEARCH_WORKER_URL` | (Optional) Worker URL used by local tooling/tests |
| `DOCSEARCH_DATA_DIR` | (Optional) directory containing source documents (defaults to `dummy-data`) |

## Deployment

1. **Deploy the Worker:**
   ```bash
   wrangler deploy
   ```

2. **Update the frontend:**
   - Edit `index.html` and update the `WORKER_URL` constant with your deployed Worker URL
   - Deploy the frontend to your preferred hosting service (GitHub Pages, Netlify, Vercel, etc.)

3. **Test the integration:**
   - Verify document search functionality
   - Test AI summarization features
   - Confirm follow-up question generation

## Project Structure

```
docsearch/
├── index.html              # Web interface for document search
├── embedding-worker.js     # Cloudflare Worker for search/AI endpoints
├── embed_files.py         # Document processing and embedding script
├── wrangler.toml           # Worker deployment configuration
├── dummy-data/            # Sample documents for testing
├── data/                  # Generated embeddings and metadata
└── README.md             # This documentation
```

---
> **This is Demo. contains no confidential data/IP**
